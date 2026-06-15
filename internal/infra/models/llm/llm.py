"""
LLM 客户端模块。
=================

基于 OpenAI SDK 的 LLM（大语言模型）服务封装，后端使用 DeepSeek API。

在记忆系统中的作用：
  1. **事实提取（extract_facts）**：从对话中提取原子化的事实陈述。
  2. **话题匹配（topic_loom）**：将新事实匹配到已有的 MemBox（记忆容器）或建议新建。
  3. **演化决策（evolution_decision）**：决定对已有记忆执行 ADD/UPDATE/MERGE/LINK 操作。
  4. **实体提取（extract_entities）**：从文本中提取命名实体。
  5. **通用生成（generate）**：底层的 LLM 调用接口，支持自定义 system prompt。

架构说明：
  - 本模块是 LLM 调用的基础设施层，Prompt 模板定义在 prompts.py 中。
  - 所有 LLM 响应均经过 JSON 解析，支持自动修复损坏的 JSON。
"""

import json
import logging

from openai import AsyncOpenAI
from json_repair import repair_json
from internal.infra.models.llm.prompts.prompts import prompts
from internal.config.settings import settings
from internal.util.api_retry import get_semaphore, with_retry
from internal.util.debug_collector import DebugCollector
from internal.util.token_tracker import tracker as token_tracker

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        # 初始化异步 OpenAI 客户端，配置 DeepSeek 作为后端
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    # 大模型生成
    async def generate(self, system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
        """
        底层 LLM 调用接口：发送 system prompt 和用户消息，返回生成的文本。

        通过全局信号量 + 指数退避重试控制大并发下的 API 调用速率。

        Args:
            system_prompt: 系统级提示（定义角色和行为约束）。
            user_message: 用户消息（具体任务输入）。
            max_tokens: 最大输出 token 数。

        Returns:
            LLM 生成的文本内容。如果响应中没有文本块，则返回空字符串。
        """
        logger.debug("LLM call: model=%s system='%s...' user='%s...'", settings.deepseek_model, system_prompt[:60], user_message[:100])
        sem = get_semaphore("llm", settings.llm_max_concurrency)

        async def _call():
            async with sem:
                return await self._client.chat.completions.create(
                    model=settings.deepseek_model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    extra_body={"reasoning_effort": "low"},
                )

        message = await with_retry(
            _call,
            max_retries=settings.api_max_retries,
            base_delay=settings.api_retry_base_delay,
            label="llm",
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            token_tracker.add(
                "llm",
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                completion=getattr(usage, "completion_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        content = message.choices[0].message.content or ""
        logger.debug("LLM response: %s", content[:300])
        return content

    # 提取事实
    async def extract_facts(self, user_input: str, event_time: str = "") -> dict:
        """
        从用户输入中提取原子化的事实。

        这是记忆管线的第一步：将自然语言对话转化为结构化的
        独立事实陈述，包含实体列表和事件时间。

        Args:
            user_input: 用户的原始对话输入。
            event_time: 对话发生的日期（用于消解相对时间引用，如"昨天"）。

        Returns:
            包含 "facts" 键的字典，每个 fact 包含 content、entities、event_time。
        """
        if event_time:
            user_input = f"[Conversation date: {event_time}]\n\n{user_input}"
        response = await self.generate(prompts.FACT_EXTRACTION_SYSTEM, user_input)
        return self._parse_json(response)

    # 话题亲和性
    async def topic_loom(self, fact_content: str, memboxes: list[dict]) -> dict:
        """
        话题匹配：将新事实分配到最合适的 MemBox。

        综合考虑语义相似度、时间邻近度、说话人重叠和主题连贯性，
        决定事实应该放入哪个已有的 MemBox，或建议创建新的 MemBox。

        Args:
            fact_content: 待分配的事实内容。
            memboxes: 已有的 MemBox 列表（序列化为字典）。

        Returns:
            包含 membox_id、new_membox_title、confidence、reasoning 的字典。
        """
        user_message = json.dumps({
            "fact": fact_content,
            "existing_memboxes": memboxes,
        }, ensure_ascii=False)
        response = await self.generate(prompts.TOPIC_LOOM_SYSTEM, user_message)
        return self._parse_json(response)

    # 做决定
    async def evolution_decision(self, fact_content: str, membox_context: dict, existing_facts: list[dict]) -> dict:
        """
        演化决策：决定新事实如何与已有记忆交互。

        四种可能的决策：
          - ADD：添加为全新事实（默认策略）。
          - UPDATE：新事实与已有事实矛盾，替换旧事实。
          - MERGE：新事实与已有事实描述完全相同的事件。
          - LINK：关联到已有事实但保持独立。

        Args:
            fact_content: 新事实内容。
            membox_context: 目标 MemBox 的上下文信息。
            existing_facts: MemBox 中已有的事实列表。

        Returns:
            包含 decision、target_fact_id、merged_content、reasoning 的字典。
        """
        user_message = json.dumps({
            "fact": fact_content,
            "membox": membox_context,
            "existing_facts": existing_facts,
        }, ensure_ascii=False)
        response = await self.generate(prompts.EVOLUTION_DECISION_SYSTEM, user_message)
        return self._parse_json(response)

    # 提取实体
    async def extract_entities(self, text: str) -> list[str]:
        response = await self.generate(prompts.ENTITY_EXTRACTION_SYSTEM, text, max_tokens=2048)
        result = self._parse_json(response)
        return result.get("entities", [])

    # 根据记忆上下文回答问题
    async def generate_answer(
        self, query: str, context: str,
        *, debug: DebugCollector | None = None,
    ) -> dict:
        """
        基于检索到的记忆上下文回答问题。

        Args:
            query: 用户问题。
            context: 来自 CBA 构建的检索上下文（含事实和最近对话）。

        Returns:
            包含 answer、confidence、reasoning 的字典。
        """
        user_message = f"问题：{query}\n\n记忆上下文：\n{context}"
        response = await self.generate(prompts.ANSWER_SYNTHESIS_SYSTEM, user_message)
        parsed = self._parse_json(response)
        if debug is not None:
            debug.record("llm_answer", {
                "system_prompt": prompts.ANSWER_SYNTHESIS_SYSTEM,
                "user_message": user_message,
                "raw_response": response,
                "parsed": parsed,
            })
        return parsed

    @staticmethod
    def _parse_json(response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed, attempting repair. Original: %s", text[:200])
            return json.loads(repair_json(text))


# 全局单例：方便在整个应用中共享同一个 LLMClient 实例
llm_client = LLMClient()

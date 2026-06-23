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

import httpx
from openai import AsyncOpenAI
from json_repair import repair_json
from internal.infra.models.llm.prompts.prompts import prompts
from internal.config.settings import settings
from internal.util.api_retry import get_semaphore, with_retry
from internal.util.call_trace import get_trace
from internal.util.debug_collector import DebugCollector
from internal.util.token_tracker import tracker as token_tracker

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        # 初始化异步 OpenAI 客户端，配置 DeepSeek 作为后端
        # SDK 层 httpx 超时：connect 5s / read=write=pool 与外层 wait_for 一致；
        # max_retries=0：避免 SDK 内部还做指数退避（我们外层 with_retry 已经做了）。
        t = settings.llm_call_timeout_s
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(t, connect=5.0),
            max_retries=0,
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
        logger.debug("LLM call: model=%s system='%s...' user='%s...'", settings.llm_model, system_prompt[:60], user_message[:100])
        sem = get_semaphore("llm", settings.llm_max_concurrency)

        async def _call():
            async with sem:
                return await self._client.chat.completions.create(
                    model=settings.llm_model,
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
            per_call_timeout=settings.llm_call_timeout_s,
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            token_tracker.add(
                "llm",
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                completion=getattr(usage, "completion_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        # 单题调用追踪：LLM API 计数 + 模型内部 CoT 原文（reasoning_content）
        if (t := get_trace()) is not None:
            t.llm_total += 1
            msg_obj = message.choices[0].message
            rc = getattr(msg_obj, "reasoning_content", None) or getattr(msg_obj, "reasoning", None) or ""
            t.llm_reasoning_texts.append(rc)
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
        if (t := get_trace()) is not None:
            t.llm_method("extract_entities")
        response = await self.generate(prompts.ENTITY_EXTRACTION_SYSTEM, text, max_tokens=2048)
        result = self._parse_json(response)
        return result.get("entities", [])

    # 根据记忆上下文回答问题
    async def generate_answer(
        self, query: str, context: str,
        *, reference_time: str | None = None,
        debug: DebugCollector | None = None,
    ) -> dict:
        """
        基于检索到的记忆上下文回答问题。

        Args:
            query: 用户问题。
            context: 来自 CBA 构建的检索上下文（含事实和最近对话）。
            reference_time: 参考时间（ISO格式），时间推理类问题以此作为"当前时刻"。

        Returns:
            包含 answer、confidence、reasoning 的字典。
        """
        # 语言提示：根据问题语言决定输出语言，避免中文上下文导致英文问题输出中文
        if (t := get_trace()) is not None:
            t.llm_method("generate_answer")
        lang_hint = "IMPORTANT: Write your answer in English." if all(ord(c) < 0x4e00 for c in query[:50]) else ""
        ref_line = f"\n参考时间（当前时刻）：{reference_time}\n" if reference_time else ""
        user_message = f"问题：{query}\n\n记忆上下文：\n{context}{ref_line}\n\n{lang_hint}"
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

    # 充分性检查
    async def check_sufficiency(
        self, query: str, context_summary: str,
        *, debug: DebugCollector | None = None,
    ) -> dict:
        """检查当前召回的上下文是否足以回答问题。

        Args:
            query: 用户问题。
            context_summary: 已召回事实的紧凑摘要。

        Returns:
            包含 sufficient、confidence、reasoning、alternative_keywords 的字典。
        """
        if (t := get_trace()) is not None:
            t.llm_method("check_sufficiency")
        user_message = json.dumps({
            "query": query,
            "retrieved_context_summary": context_summary,
        }, ensure_ascii=False)
        response = await self.generate(
            prompts.SUFFICIENCY_CHECK_SYSTEM, user_message, max_tokens=512,
        )
        parsed = self._parse_json(response)
        if debug is not None:
            debug.record("sufficiency_check", {
                "user_message": user_message,
                "raw_response": response,
                "parsed": parsed,
            })
        return parsed

    # 反向实体过滤：从 query 推断"完整答案应当引用的具体实体"，检查 top-N 召回是否包含
    async def extract_expected_entities(
        self, query: str, top_facts_summary: str,
        *, debug: DebugCollector | None = None,
    ) -> dict:
        """抽取问题预期需要引用的具体实体，并标记 top 召回中缺失的实体。

        Args:
            query: 用户问题。
            top_facts_summary: top-N MAS facts 的紧凑摘要。

        Returns:
            {"expected_entities": [...], "missing_in_top": [...]}
        """
        if (t := get_trace()) is not None:
            t.llm_method("extract_expected_entities")
        user_message = json.dumps({
            "query": query,
            "top_retrieved_facts": top_facts_summary,
        }, ensure_ascii=False)
        response = await self.generate(
            prompts.EXPECTED_ENTITY_SYSTEM, user_message, max_tokens=512,
        )
        parsed = self._parse_json(response)
        if debug is not None:
            debug.record("expected_entities", {
                "user_message": user_message,
                "raw_response": response,
                "parsed": parsed,
            })
        return parsed

    # 多智能体辩论回答
    async def debate_answer(
        self, query: str, context: str,
        *, reference_time: str | None = None,
        debug: DebugCollector | None = None,
    ) -> dict:
        """多专家视角回答 + Judge 合并。

        并行调用 3 个 specialist（preference / factual / temporal），
        再由 judge 综合输出最终答案。

        Args:
            query: 用户问题。
            context: 来自 CBA 的检索上下文。
            reference_time: 参考时间（ISO格式）。

        Returns:
            包含 answer、confidence、reasoning 的字典。
        """
        if (t := get_trace()) is not None:
            t.llm_method("debate_answer")
        import asyncio

        lang_hint = "IMPORTANT: Write your answer in English." if all(ord(c) < 0x4e00 for c in query[:50]) else ""
        ref_line = f"\n参考时间（当前时刻）：{reference_time}\n" if reference_time else ""
        user_message = f"问题：{query}\n\n记忆上下文：\n{context}{ref_line}\n\n{lang_hint}"

        specialist_prompts = [
            prompts.PREFERENCE_SUMMARIZER_SYSTEM,
            prompts.FACTUAL_RECALLER_SYSTEM,
            prompts.TEMPORAL_REASONER_SYSTEM,
        ]

        # 并行调用 specialist
        specialist_tasks = [
            self.generate(sp, user_message)
            for sp in specialist_prompts
        ]
        specialist_responses = await asyncio.gather(*specialist_tasks, return_exceptions=True)

        specialist_answers = []
        for i, resp in enumerate(specialist_responses):
            if isinstance(resp, Exception):
                logger.warning("Specialist %d failed: %s", i, resp)
                specialist_answers.append({"answer": "", "confidence": 0.0, "reasoning": f"specialist {i} failed"})
            else:
                specialist_answers.append(self._parse_json(resp))

        # Judge 合并
        judge_input = json.dumps({
            "query": query,
            "specialist_answers": [
                {"role": name, "answer": ans.get("answer", ""), "confidence": ans.get("confidence", 0),
                 "reasoning": ans.get("reasoning", "")}
                for name, ans in zip(
                    ["preference_summarizer", "factual_recaller", "temporal_reasoner"],
                    specialist_answers,
                )
            ],
        }, ensure_ascii=False)

        judge_response = await self.generate(prompts.DEBATE_JUDGE_SYSTEM, judge_input)
        parsed = self._parse_json(judge_response)

        if debug is not None:
            debug.record("debate_answer", {
                "specialist_answers": specialist_answers,
                "judge_raw": judge_response,
                "judge_parsed": parsed,
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

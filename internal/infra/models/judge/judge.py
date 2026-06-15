"""Judge LLM 客户端：用于评测打分。

仿照 internal/infra/models/llm/llm.py 的结构，使用 Anthropic AsyncAnthropic
SDK，支持自定义 base_url / api_key / model（适配兼容 Anthropic 协议的网关）。

复用 internal/util/api_retry 的全局信号量 + 指数退避，避免打分大批量并发时
撞速率限制。

打分语义：返回字符串答复（"yes" / "no"），由调用方解析。
"""

from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from internal.config.settings import settings
from internal.util.api_retry import get_semaphore, with_retry
from internal.util.token_tracker import tracker as token_tracker

logger = logging.getLogger(__name__)


class JudgeClient:
    """评测打分用的 LLM 客户端（Anthropic 协议）。"""

    def __init__(self) -> None:
        if not settings.judge_api_key or not settings.judge_base_url or not settings.judge_model:
            logger.warning(
                "JudgeClient: settings.judge_api_key / judge_base_url / judge_model "
                "尚未配置，调用前请先填写。"
            )
        self._client = AsyncAnthropic(
            api_key=settings.judge_api_key or "EMPTY",
            base_url=settings.judge_base_url or None,
        )

    async def judge(self, prompt: str, max_tokens: int = 10) -> str:
        """以"yes/no"为期望输出运行打分。返回原始文本（已 strip）。"""
        sem = get_semaphore("judge", settings.llm_max_concurrency)

        async def _call():
            async with sem:
                return await self._client.messages.create(
                    model=settings.judge_model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )

        message = await with_retry(
            _call,
            max_retries=settings.api_max_retries,
            base_delay=settings.api_retry_base_delay,
            label="judge",
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            inp = getattr(usage, "input_tokens", 0) or 0
            outp = getattr(usage, "output_tokens", 0) or 0
            token_tracker.add("judge", prompt=inp, completion=outp, total=inp + outp)
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        return ""


judge_client = JudgeClient()

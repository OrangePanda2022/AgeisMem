"""
Embedding 客户端模块。
======================

基于火山引擎 Ark API 的多模态 Embedding 服务封装。

功能：
  1. 将文本转换为向量嵌入（embedding），用于语义相似度计算。
  2. 支持批量文本嵌入（逐条调用，串行执行）。
  3. 使用 "doubao-embedding-vision-251215" 模型，该模型同时支持文本和图像的嵌入。

在记忆系统中的作用：
  - 输入阶段：对提取的事实进行嵌入，存入向量数据库。
  - 检索阶段：将用户查询转为嵌入向量，进行语义相似度搜索。
  - MAS 计算阶段：计算查询嵌入与事实嵌入的余弦相似度。

使用方式：
  >>> from internal.infrastructure.models.embedding.embedding import embedding_client
  >>> vec = await embedding_client.embed_text("一段文本")
"""

import logging

import httpx
from volcenginesdkarkruntime import AsyncArk

from internal.config.settings import settings
from internal.util.api_retry import get_semaphore, with_retry
from internal.util.token_tracker import tracker as token_tracker

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Embedding 客户端：封装火山引擎 Ark 的多模态 Embedding API。

    提供文本到向量嵌入的转换能力，是记忆系统语义检索的基础组件。
    """

    def __init__(self) -> None:
        # SDK 默认 read=600s + max_retries=2，会和外层 with_retry 串成天文超时；
        # 这里压到 timeout=embedding_call_timeout_s + max_retries=0，由外层兜底。
        t = settings.embedding_call_timeout_s
        self._client = AsyncArk(
            api_key=settings.ark_api_key,
            timeout=httpx.Timeout(t, connect=5.0),
            max_retries=0,
        )
        # 使用豆包 Embedding Vision 模型，支持文本和图像嵌入（当前仅使用文本功能）
        self._model = "doubao-embedding-vision-251215"

    async def embed_text(self, text: str) -> list[float]:
        logger.debug("Embedding: text='%s...'", text[:80])
        sem = get_semaphore("embedding", settings.embedding_max_concurrency)

        async def _call():
            async with sem:
                return await self._client.multimodal_embeddings.create(
                    model=self._model,
                    input=[{"type": "text", "text": text}],
                )

        resp = await with_retry(
            _call,
            max_retries=settings.api_max_retries,
            base_delay=settings.api_retry_base_delay,
            label="embedding",
            per_call_timeout=settings.embedding_call_timeout_s,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            token_tracker.add(
                "embedding",
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        embedding = resp.data.embedding
        logger.debug("Embedding: dim=%d", len(embedding))
        return embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        tasks = [self.embed_text(t) for t in texts]
        results = []
        for task in tasks:
            results.append(await task)
        return results


# 全局单例：方便在整个应用中共享同一个 EmbeddingClient 实例
embedding_client = EmbeddingClient()

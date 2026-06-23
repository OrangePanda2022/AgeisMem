"""
Embedding 客户端模块。
======================

基于 OpenAI Embeddings API 的文本向量服务封装。

功能：
  1. 将文本转换为向量嵌入（embedding），用于语义相似度计算。
  2. 支持批量文本嵌入（逐条调用，串行执行）。

在记忆系统中的作用：
  - 输入阶段：对提取的事实进行嵌入，存入向量数据库。
  - 检索阶段：将用户查询转为嵌入向量，进行语义相似度搜索。
  - MAS 计算阶段：计算查询嵌入与事实嵌入的余弦相似度。

使用方式：
  >>> from internal.infra.models.embedding.embedding import embedding_client
  >>> vec = await embedding_client.embed_text("一段文本")
"""

import asyncio
import logging

import httpx
from openai import AsyncOpenAI

from internal.config.settings import settings
from internal.util.api_retry import get_semaphore, with_retry
from internal.util.call_trace import get_trace
from internal.util.token_tracker import tracker as token_tracker

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Embedding 客户端：封装 OpenAI Embeddings API。

    提供文本到向量嵌入的转换能力，是记忆系统语义检索的基础组件。
    """

    def __init__(self) -> None:
        t = settings.embedding_call_timeout_s
        self._client = AsyncOpenAI(
            api_key=settings.embedding_api_key or "EMPTY",
            base_url=settings.embedding_base_url or None,
            timeout=httpx.Timeout(t, connect=5.0),
            max_retries=0,
        )
        # 进程内向量缓存：以 text 为 key。ingest 阶段关键词/fact 内容跨消息
        # 大量重复（如"特斯拉""用户""2023"），缓存后同一进程内只 embed 一次。
        # 单飞（singleflight）：同一 text 并发请求时只打一次 API，其余等 Future。
        self._cache: dict[str, list[float]] = {}
        self._inflight: dict[str, asyncio.Future[list[float]]] = {}
        self._cache_max = settings.embedding_cache_max

    def _cache_key(self, text: str) -> str:
        """key 含 model+dim，换配置时缓存自然失效。"""
        return f"{settings.embedding_model}\x00{settings.embedding_dim}\x00{text}"

    async def _embed_one(self, text: str) -> list[float]:
        """单条 embedding，带缓存 + 单飞去重。"""
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            if (tr := get_trace()) is not None:
                tr.embedding_cache_hits += 1
            return cached
        fut = self._inflight.get(key)
        if fut is None:
            fut = asyncio.get_event_loop().create_future()
            self._inflight[key] = fut
            try:
                vec = await self._embed_one_uncached(text)
                if len(self._cache) >= self._cache_max:
                    # 简单 FIFO 淘汰：弹出最早插入的 key
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = vec
                fut.set_result(vec)
            except BaseException as e:
                fut.set_exception(e)
                raise
            finally:
                self._inflight.pop(key, None)
        return await fut

    async def _embed_one_uncached(self, text: str) -> list[float]:
        if (tr := get_trace()) is not None:
            tr.embedding_api_calls += 1
            tr.embedding_texts += 1
        sem = get_semaphore("embedding", settings.embedding_max_concurrency)

        async def _call():
            async with sem:
                return await self._client.embeddings.create(
                    model=settings.embedding_model,
                    input=text,
                    dimensions=settings.embedding_dim,
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
        embedding = resp.data[0].embedding
        logger.debug("Embedding: dim=%d", len(embedding))
        return embedding

    async def embed_text(self, text: str) -> list[float]:
        """单条 embedding（命中缓存时不打 API、不占限流槽）。"""
        return await self._embed_one(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding：逐 text 走缓存，未命中的才合并打一次 API。"""
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(self._cache_key(t))
            if cached is not None:
                if (tr := get_trace()) is not None:
                    tr.embedding_cache_hits += 1
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            fresh = await self._embed_batch_uncached(miss_texts)
            for i, vec in zip(miss_idx, fresh):
                results[i] = vec
                key = self._cache_key(texts[i])
                if len(self._cache) >= self._cache_max:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = vec
        return [r for r in results if r is not None]

    async def _embed_batch_uncached(self, texts: list[str]) -> list[list[float]]:
        if (tr := get_trace()) is not None:
            tr.embedding_api_calls += 1
            tr.embedding_texts += len(texts)
        sem = get_semaphore("embedding", settings.embedding_max_concurrency)

        async def _call():
            async with sem:
                return await self._client.embeddings.create(
                    model=settings.embedding_model,
                    input=texts,
                    dimensions=settings.embedding_dim,
                )

        resp = await with_retry(
            _call,
            max_retries=settings.api_max_retries,
            base_delay=settings.api_retry_base_delay,
            label="embedding_batch",
            per_call_timeout=settings.embedding_call_timeout_s,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            token_tracker.add(
                "embedding",
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        # resp.data 已按 input index 排序
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


# 全局单例：方便在整个应用中共享同一个 EmbeddingClient 实例
embedding_client = EmbeddingClient()

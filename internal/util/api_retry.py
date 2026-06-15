"""外部 API 调用的限流 + 重试小工具。

懒初始化全局 Semaphore（按当前事件循环），并提供指数退避重试包装。
不引入第三方依赖（避免 tenacity / aiolimiter）。
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_semaphores: dict[str, asyncio.Semaphore] = {}


def get_semaphore(name: str, max_concurrency: int) -> asyncio.Semaphore:
    """按名字返回模块级 Semaphore；首次访问时创建。

    在事件循环外不会创建，等到首个协程调用时才会绑定到当前 loop。
    """
    sem = _semaphores.get(name)
    if sem is None:
        sem = asyncio.Semaphore(max_concurrency)
        _semaphores[name] = sem
    return sem


async def with_retry(
    call: Callable[[], Awaitable[T]],
    max_retries: int,
    base_delay: float,
    label: str = "api",
) -> T:
    """指数退避重试。仅捕获网络/限速类异常，业务解析错误不重试。

    捕获策略：所有 Exception 都重试，但最后一次直接 raise。
    base_delay * 2**i + jitter(0, base_delay)。
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
            logger.warning(
                "%s call failed (attempt %d/%d): %s; retrying in %.2fs",
                label, attempt + 1, max_retries + 1, e, delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc

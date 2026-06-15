"""
事实（Fact）仓储接口模块。

定义事实持久化操作的抽象接口，包括增删改查、按记忆盒子查询、
向量相似度搜索及批量操作等功能。
"""

from abc import ABC, abstractmethod
from uuid import UUID

from internal.domain.model.fact import Fact


class FactRepository(ABC):
    """事实仓储抽象基类。
    """

    @abstractmethod
    async def add(self, fact: Fact) -> None:
        """新增一个事实到持久层。"""
        ...

    @abstractmethod
    async def get_by_id(self, fact_id: UUID) -> Fact | None:
        """根据事实 ID 查询事实，不存在时返回 None。"""
        ...

    @abstractmethod
    async def find_by_membox_id(self, membox_id: UUID) -> list[Fact]:
        """根据记忆盒子 ID 查找该盒子下所有事实。"""
        ...

    @abstractmethod
    async def vector_search(self, embedding: list[float], top_k: int = 20) -> list[Fact]:
        """向量相似度搜索：根据嵌入向量查找最相似的 top_k 个事实。"""
        ...

    @abstractmethod
    async def update(self, fact: Fact) -> None:
        """更新一个已有事实的信息。"""
        ...

    @abstractmethod
    async def delete(self, fact_id: UUID) -> None:
        """删除指定事实。"""
        ...

    @abstractmethod
    async def batch_add(self, facts: list[Fact]) -> None:
        """批量新增多个事实。"""
        ...

    async def get_by_ids(self, fact_ids: list[UUID]) -> list[Fact]:
        """批量根据 ID 获取事实列表。

        默认实现逐个调用 get_by_id。子类可重写以使用批量查询优化。
        """
        result: list[Fact] = []
        for fid in fact_ids:
            f = await self.get_by_id(fid)
            if f:
                result.append(f)
        return result

    async def vector_search_with_scores(self, embedding: list[float], top_k: int = 20) -> list[tuple[Fact, float]]:
        """带相似度分数的向量搜索。

        与 vector_search 类似，但返回 (事实, 余弦相似度) 元组列表。
        默认实现调用 vector_search()，并将分数设为 0.0。
        子类应重写以返回真实的相似度分数。
        """
        facts = await self.vector_search(embedding, top_k)
        return [(f, 0.0) for f in facts]

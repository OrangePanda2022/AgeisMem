"""
实体（Entity）仓储接口模块。

定义实体持久化操作的抽象接口，包括增删改查、按名称查找、向量相似度搜索等功能。
"""

from abc import ABC, abstractmethod
from uuid import UUID

from internal.domain.model.entity import Entity


class EntityRepository(ABC):
    """实体仓储抽象基类。
    """

    @abstractmethod
    async def add(self, entity: Entity) -> None:
        """新增一个实体到持久层。"""
        ...

    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Entity | None:
        """根据实体 ID 查询实体，不存在时返回 None。"""
        ...

    @abstractmethod
    async def get_by_name(self, name: str) -> Entity | None:
        """根据实体名称精确查询实体，不存在时返回 None。"""
        ...

    @abstractmethod
    async def find_similar(self, embedding: list[float], top_k: int = 10) -> list[Entity]:
        """向量相似度搜索：根据嵌入向量查找最相似的 top_k 个实体。"""
        ...

    @abstractmethod
    async def upsert_by_name(self, entity: Entity) -> Entity:
        """按名称执行 upsert：若实体已存在则更新，否则新增。返回最终的实体对象。"""
        ...

    @abstractmethod
    async def find_by_names(self, names: list[str]) -> list[Entity]:
        """批量按名称查询实体列表。"""
        ...

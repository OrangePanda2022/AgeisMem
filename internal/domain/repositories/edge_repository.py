"""
边仓储接口模块。

定义实体、事实和边之间关联关系的持久化操作抽象接口。
提供图遍历功能，从一组事实出发沿着实体边查找关联事实。
"""

from abc import ABC, abstractmethod
from uuid import UUID

from internal.domain.model.edge import Edge


class EdgeRepository(ABC):
    """边仓储抽象基类。

    管理事实之间的边，提供增删改查及图遍历功能。
    具体实现应对接底层数据库（如图数据库或关系数据库）。
    """

    @abstractmethod
    async def add(self, edge: Edge) -> None:
        """添加一条新边到持久层。"""
        ...

    @abstractmethod
    async def find_by_fact(self, fact_id: UUID) -> list[Edge]:
        """根据事实 ID 查找所有关联的边。"""
        ...

    @abstractmethod
    async def update_weight(self, edge_id: UUID, weight: float) -> None:
        """更新指定边的权重值。"""
        ...

    @abstractmethod
    async def delete_by_fact(self, fact_id: UUID) -> None:
        """删除与指定事实关联的所有边。"""
        ...

    # TODO
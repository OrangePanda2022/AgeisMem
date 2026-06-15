"""
记忆盒子（MemBox）仓储接口模块。

定义记忆盒子持久化操作的抽象接口，包括增删改查、按时间范围和分层等级查询等功能。
"""

from abc import ABC, abstractmethod
from uuid import UUID

from internal.domain.model.membox import MemBox


class MemBoxRepository(ABC):
    """记忆盒子仓储抽象基类。

    提供记忆盒子的完整 CRUD 操作接口，支持按 ID、时间范围、分层等级进行检索。
    具体实现应对接底层数据库（如文档数据库或关系数据库）。
    """

    @abstractmethod
    async def add(self, membox: MemBox) -> None:
        """新增一个记忆盒子到持久层。"""
        ...

    @abstractmethod
    async def get_by_id(self, membox_id: UUID) -> MemBox | None:
        """根据记忆盒子 ID 查询，不存在时返回 None。"""
        ...

    @abstractmethod
    async def find_by_time_range(self, start: str | None, end: str | None, limit: int = 100) -> list[MemBox]:
        """按事件时间范围查找记忆盒子列表。

        参数 start 和 end 为 ISO 格式时间字符串，None 表示不限边界。
        """
        ...

    @abstractmethod
    async def find_by_tier(self, tier: str, limit: int = 100) -> list[MemBox]:
        """按存储分层等级查找记忆盒子列表。"""
        ...

    @abstractmethod
    async def list_all(self, offset: int = 0, limit: int = 50) -> list[MemBox]:
        """分页列出所有记忆盒子。"""
        ...

    @abstractmethod
    async def update(self, membox: MemBox) -> None:
        """更新一个已有记忆盒子的信息。"""
        ...

    @abstractmethod
    async def delete(self, membox_id: UUID) -> None:
        """删除指定记忆盒子。"""
        ...

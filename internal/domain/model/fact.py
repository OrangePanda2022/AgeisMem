"""
事实（Fact）数据类模块。

定义事实的核心数据结构。事实是记忆盒子（MemBox）中的基本记忆单元，
包含文本内容、向量嵌入、评分和分层等级等信息。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

from internal.domain.model.tier import Tier
from internal.domain.model.metadata import Metadata
from internal.domain.model.tag import Tag

@dataclass
class Fact:
    """事实数据类：表示一条原子记忆/事实。

    每个事实属于一个记忆盒子，包含文本内容、向量嵌入、相关度评分和分层等级。
    支持通过 bump_score 提升评分、migrate_tier 迁移分层，以及 record_access 记录访问。
    """

    # 事实的唯一标识符
    id: UUID = field(default_factory=uuid4)
    # 所属记忆盒子的 ID
    membox_id: UUID | None = None
    # 事实的文本内容
    content: str = ""
    # 事实的向量嵌入表示
    embedding: list[float] | None = None
    # 相关度/重要性评分，范围 [0, 1]
    score: float = 0.0
    # 存储分层等级（L0 活跃 / L1 温 / L2 冷 / L3 归档）
    tier: Tier = Tier.L0
    # 创建时间（UTC）
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 最后访问时间（UTC），用于计算热度和衰减
    last_accessed_at: datetime | None = None
    # 累计访问次数
    access_count: int = 0
    # 消息原文
    original_msg: str = ""
    # 附加元数据
    metadata: Metadata = field(default_factory=Metadata)
    # 标签
    tag: List[Tag] = field(default_factory=list)

    def bump_score(self, increment: float = 0.1) -> None:
        """提升事实的评分分数。"""
        self.score = min(1.0, self.score + increment)
        self.record_access()

    def migrate_tier(self, new_tier: Tier) -> None:
        """将事实迁移到新的存储分层等级。"""
        self.tier = new_tier

    def record_access(self) -> None:
        """记录对事实的访问：更新最后访问时间并递增访问计数。"""
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1

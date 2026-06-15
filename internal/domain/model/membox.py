"""
记忆盒子（MemBox）数据类模块。

定义记忆盒子的核心数据结构。记忆盒子是记忆管理中的顶层容器，
将一组相关的事实（Fact）聚合在一起，并维护综合评分和分层等级。
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

from internal.domain.model.fact import Fact

@dataclass
class MemBox:
    """记忆盒子数据类：一组相关事实的容器。

    每个记忆盒子有标题、摘要、综合评分和分层等级。
    综合评分（box_score）由访问频率、最近使用、实体中心度和用户兴趣四部分加权计算。
    """

    # 记忆盒子的唯一标识符
    id: UUID = field(default_factory=uuid4)
    # 记忆盒子标题
    title: str = ""
    # 记忆盒子内容摘要
    summary: str = ""
    # 综合评分（box_score），由多因素加权计算
    box_score: float = 0.0
    # 创建时间（UTC）
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 最后更新时间（UTC）
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 最后访问时间（UTC）
    last_accessed_at: datetime | None = None
    # 累计访问次数
    access_count: int = 0
    # 盒中有什么事实
    content: List[Fact] | None = None

    def bump_score(self, increment: float = 0.1) -> None:
        """提升记忆盒子的综合评分。

        将 box_score 增加指定增量，上限为 1.0，同时记录本次访问。
        """
        self.box_score = min(1.0, self.box_score + increment)
        self.record_access()

    def compute_box_score(self, centrality: float = 0.0, user_interest_delta: float = 0.0) -> float:
        """计算综合评分 BoxScore：综合考虑访问频率、最近使用、实体中心度和用户兴趣。

        加权公式: 0.3 * 访问频率 + 0.2 * 最近使用衰减 + 0.2 * 实体中心度 + 0.3 * (当前评分 + 用户兴趣增量)

        返回计算后的 box_score 值（范围 [0, 1]）。
        """
        # 访问频率：访问次数 / 50，上限 1.0
        access_freq = min(1.0, self.access_count / 50.0)

        # 最近使用衰减：基于最后访问时间的天数，使用指数衰减
        recency = 0.5
        if self.last_accessed_at is not None:
            days = (datetime.now(timezone.utc) - self.last_accessed_at).total_seconds() / 86400.0
            recency = math.exp(-0.01 * days)

        # 加权融合：0.3*访问频率 + 0.2*最近使用 + 0.2*中心度 + 0.3*用户兴趣
        new_score = (
            0.3 * access_freq
            + 0.2 * recency
            + 0.2 * centrality
            + 0.3 * (self.box_score + user_interest_delta)
        )
        return min(1.0, max(0.0, new_score))

    def update_summary(self, summary: str) -> None:
        """更新记忆盒子的摘要信息，同时刷新更新时间戳。"""
        self.summary = summary
        self.updated_at = datetime.now(timezone.utc)

    def record_access(self) -> None:
        """记录对记忆盒子的访问：更新最后访问时间并递增访问计数。"""
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1

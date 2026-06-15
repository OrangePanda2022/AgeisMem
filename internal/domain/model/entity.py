"""
实体（Entity）数据类模块。

定义命名实体的核心数据结构。实体是知识图谱中的节点，
包含名称、向量嵌入和中心度等属性，用于表示对话或文档中的关键概念。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4


@dataclass
class Entity:
    """实体数据类：表示一个命名实体（人物、地点、概念等）。

    每个实体拥有唯一标识符、名称、向量嵌入和中心度分数。
    中心度用于衡量该实体在图谱中的重要程度。
    """

    # 实体的唯一标识符
    id: UUID = field(default_factory=uuid4)
    # 实体名称（例如人名、地名、组织名等）
    name: str = ""
    # 实体的向量嵌入表示，用于语义相似度计算
    embedding: list[float] | None = None
    # 中心度分数，表示实体在图谱中的重要程度
    centrality: float = 0.0
    # 创建时间（UTC）
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 最后更新时间（UTC）
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

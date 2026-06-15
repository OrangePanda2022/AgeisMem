"""
边（Edge）数据类模块。

定义连接实体（Entity）与事实（Fact）的边数据结构。
边表示实体与事实之间的关联关系，包含权重信息，用于构建实体-事实图谱。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import List

class Category:
    MENTIONS: str = "MENTIONS"
    WORKS_AT: str = "WORKS_AT"
    LOCATED_IN: str = "LOCATED_IN"
    PART_OF: str = "PART_OF"
    CAUSED: str = "CAUSED"
    CONTRADICTS: str = "CONTRADICTS"
    PREFERS: str = "PREFERS"
    DERIVED_FROM: str = "DERIVED_FROM"
    RELATED_TO: str = "RELATED_TO"

@dataclass
class Edge:
    """边数据类：连接一个实体和一个事实，表示它们之间的语义关联。

    每一条边记录了一个实体与一个事实之间的关联强度（weight），
    用于知识图谱中的图遍历、相关性计算等场景。
    """

    # 边的唯一标识符
    id: UUID = field(default_factory=uuid4)
    # 关联的事实 ID
    from_fact_id: UUID = field(default_factory=uuid4)
    to_fact_id: UUID = field(default_factory=uuid4)
    # 关联类型
    info: Category = field(default_factory=Category)
    # 关联置信度，范围 [0, 1]，默认 0.5
    weight: float = 0.5
    # 创建时间（UTC）
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 最后更新时间（UTC）
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 历史关联类型
    history: List[Category] | None = None

"""
值对象（Value Object）模块。

定义领域层中使用的枚举类型和简单数据载体。
"""

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

class NoveltyGateResult(str, Enum):
    """新颖性门控判定结果枚举。

    NOOP  - 无操作（信息已存在，无需处理）
    ADD   - 新增（信息是全新的）
    VAGUE - 模糊（信息不够明确，需要进一步判断）
    """
    NOOP = "NOOP"
    ADD = "ADD"
    VAGUE = "VAGUE"


class EvolutionDecision(str, Enum):
    """知识演化决策枚举。

    当新信息到达时系统对现有知识做出的演化操作：
    ADD    - 新增一条独立知识
    UPDATE - 更新现有知识
    MERGE  - 合并到现有知识中
    LINK   - 与现有知识建立关联
    NOOP   - 无操作
    """
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    LINK = "LINK"
    NOOP = "NOOP"


@dataclass
class EvolutionPlan:
    """演化计划数据类：描述对知识库进行一次演化的具体方案。

    包含演化决策类型、目标记忆盒子和事实的 ID、变更内容以及新记忆盒子的标题。
    """
    # 演化决策类型
    decision: EvolutionDecision
    # 目标记忆盒子 ID
    target_membox_id: UUID | None = None
    # 目标事实 ID
    target_fact_id: UUID | None = None
    # 变更内容字典
    # TODO 这是什么？
    changes: dict = field(default_factory=dict)
    # 新记忆盒子的标题（当需要创建新盒子时使用）
    new_membox_title: str | None = None

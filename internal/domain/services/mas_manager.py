"""
记忆激活分数 (MAS) 管理系统 — v2。

MAS_fact = w1*SemanticMatch + w2*EdgeWeight + w3*Recency
         + w4*TierBoost + w5*ActivationHistory

五个评分维度：
  1. SemanticMatch (语义匹配):  查询向量与 Fact 向量的余弦相似度
  2. EdgeWeight  (边权重):      知识图谱中该 Fact 的连通性
  3. Recency     (时间近度):    基于故事事件时间的指数衰减
  4. TierBoost   (层级加成):    根据记忆层级 (L0-L3) 赋予不同权重
  5. ActivationHistory (激活历史): 基于历史访问次数的归一化分数
"""

import logging
import math
from datetime import datetime, timezone

import numpy as np

from internal.config.settings import settings
from internal.domain.model.fact import Fact
from internal.domain.model.membox import MemBox
from internal.domain.model.tier import Tier

logger = logging.getLogger(__name__)

# 各层级的加权系数：L0 核心记忆不加权衰减，L3 近乎遗忘的记忆权重最低
_TIER_BOOST = {Tier.L0: 1.0, Tier.L1: 0.7, Tier.L2: 0.4, Tier.L3: 0.1}


class MASManager:
    """记忆激活分数 (Memory Activation Score) 计算器 — v2。

    综合五个维度计算每个 Fact 与查询的相关性分数：

    MAS_fact = w1*SemanticMatch + w2*EdgeWeight + w3*Recency
             + w4*TierBoost + w5*ActivationHistory
    """

    def __init__(self) -> None:
        # 各维度的权重从配置文件加载
        self._weights = settings.mas_weights

    def compute_fact_mas(
        self,
        query_embedding: list[float],
        fact: Fact,
        *,
        membox_event_time: datetime | None = None,
        edge_weight: float = 0.0,
        reference_time: datetime | None = None,
        return_breakdown: bool = False,
    ):
        """计算单个 Fact 的综合 MAS 分数。

        参数:
            query_embedding: 查询文本的向量嵌入。
            fact: 待评分的 Fact 对象。
            membox_event_time: 关联 MemBox 的事件时间（用于计算时间近度）。
            edge_weight: 知识图谱中的归一化边权重。
            reference_time: 计算 recency 的"现在"参考点；评测时传入数据集
                最新事件时间，避免墙钟差异让全量 fact 衰减归零。
            return_breakdown: 若为 True，返回 (mas, dict) 包含五个子项；默认只返回 float。

        返回:
            综合 MAS 分数（浮点数），或 (float, dict)。
        """
        # w1: SemanticMatch — 查询向量与 Fact 向量的余弦相似度
        semantic_match = self._cosine_sim(query_embedding, fact.embedding)

        # w2: EdgeWeight — 知识图谱连通性（由调用方提供归一化值）
        # w3: Recency — 基于故事事件时间的指数衰减
        recency = self._compute_recency(membox_event_time, reference_time=reference_time)

        # w4: TierBoost — 记忆层级加成
        tier_boost = _TIER_BOOST.get(fact.tier, 0.1)

        # w5: ActivationHistory — 归一化访问次数（最多 20 次封顶）
        activation = min(1.0, fact.access_count / 20.0)

        # 加权求和得到最终 MAS 分数
        w = self._weights
        mas = (
            w["semantic_match"] * semantic_match
            + w["edge_weight"] * edge_weight
            + w["recency"] * recency
            + w["tier_boost"] * tier_boost
            + w["activation_history"] * activation
        )

        logger.debug(
            "Fact MAS %s: sim=%.3f edge=%.3f rec=%.3f tier=%.3f act=%.3f → %.4f",
            str(fact.id)[:8],
            semantic_match, edge_weight, recency, tier_boost, activation, mas,
        )
        if return_breakdown:
            return mas, {
                "semantic_match": float(semantic_match),
                "edge_weight": float(edge_weight),
                "recency": float(recency),
                "tier_boost": float(tier_boost),
                "activation": float(activation),
            }
        return mas

    @staticmethod
    def _cosine_sim(query_embedding: list[float], fact_embedding: list[float] | None) -> float:
        """计算查询向量与 Fact 向量之间的余弦相似度。

        如果 Fact 没有嵌入向量，返回 0.0。
        """
        if fact_embedding is None:
            return 0.0
        query_vec = np.array(query_embedding)
        fact_vec = np.array(fact_embedding)
        norm = np.linalg.norm(query_vec) * np.linalg.norm(fact_vec)
        if norm > 0:
            return float(np.dot(query_vec, fact_vec) / norm)
        return 0.0

    @staticmethod
    def _compute_recency(
        event_time: datetime | None,
        *,
        reference_time: datetime | None = None,
    ) -> float:
        """基于故事事件时间（非系统访问时间）的指数衰减。

        参数:
            event_time: 故事中的事件发生时间。
            reference_time: 用作"现在"的参考点；缺省回落到 datetime.now(utc)。
                评测时传入数据集最新事件时间，避免墙钟差异让全量 fact recency≈0。

        返回:
            1.0（参考点当天的事件），随时间衰减趋近于 0。
            半衰期约 231 天（衰减系数 0.003）。
        """
        if event_time is None:
            return 0.5  # 未知时间给予中性分数
        # 确保时区感知以进行减法运算
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        now = reference_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - event_time
        days = delta.total_seconds() / 86400.0
        # 较温和的衰减：半衰期约 231 天
        return math.exp(-0.003 * days)

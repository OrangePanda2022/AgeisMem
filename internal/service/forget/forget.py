"""拓扑感知遗忘服务：通过层级迁移实现记忆衰减。

遗忘/保留模型：
    R_i(t) = exp(-Δt / (S_i * (1 + λ * C_deg(i) * C_eigen(i))))

参数说明：
    - S_i: 记忆强度（当前分数，钳位到 [0.01, 1.0]）
    - C_deg:  归一化度中心性
    - C_eigen: 特征向量中心性近似（从度出发进行 1 次幂迭代）
    - λ: 拓扑保护权重

核心思想：连接良好、处于重要位置的记忆遗忘速度更慢。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from internal.config.settings import settings
from internal.domain.model.tier import Tier
from internal.infra.container import Container
from internal.util.debug_collector import DebugCollector

logger = logging.getLogger(__name__)


class ForgettingService:
    def __init__(self, container: Container) -> None:
        self.c = container
        self._lam = settings.topology_decay_lambda
        self._thresholds = settings.forgetting_tier_thresholds

    async def run_forgetting_cycle(
        self,
        reference_time: datetime | None = None,
        *,
        debug: DebugCollector | None = None,
    ) -> dict:
        """对全部 Fact 执行一轮拓扑感知 Ebbinghaus 衰减 + tier 迁移。

        Args:
            reference_time: 用于计算 delta_days 的"现在"。默认 datetime.now(utc)。
                评测场景下可传入数据集的最新事件时间，避免墙钟与历史
                数据时间差导致全量 fact 的 retention 归零。
        """
        facts = await self.c.facts.list_all(limit=10_000)
        if not facts:
            return {"updated": 0, "tier_changes": 0}

        # 度统计
        deg_map: dict[str, int] = {}
        for f in facts:
            deg_map[str(f.id)] = await self.c.edges.degree(f.id)
        max_deg = max(deg_map.values()) if deg_map else 1
        max_deg = max(max_deg, 1)

        # 1 次幂迭代近似 eigen-centrality：x'(i) = sum_{j in N(i)} deg(j)
        eig_map: dict[str, float] = {}
        for f in facts:
            edges = await self.c.edges.find_by_fact(f.id)
            s = 0.0
            for e in edges:
                other = e.to_fact_id if str(e.from_fact_id) == str(f.id) else e.from_fact_id
                s += deg_map.get(str(other), 0)
            eig_map[str(f.id)] = s

        max_eig = max(eig_map.values()) if eig_map else 1.0
        max_eig = max(max_eig, 1.0)

        updated = 0
        tier_changes = 0
        now = reference_time or datetime.now(timezone.utc)
        for f in facts:
            ref = f.last_accessed_at or f.created_at
            delta_days = (now - ref).total_seconds() / 86400.0
            # 防御负值：ref_time 是 haystack 最大日期（如 2023），但 ingest 期间
            # 如果某条 fact 漏了 event_time、用了 datetime.now()（如 2026），
            # delta_days 会变成大负数，下面 exp(-neg/small) 直接 OverflowError。
            # 负值代表 fact 比参考时间还"新"，按 0 处理（retention=1.0）。
            delta_days = max(0.0, delta_days)
            s_i = max(0.01, min(1.0, f.score))
            c_deg = deg_map[str(f.id)] / max_deg
            c_eig = eig_map[str(f.id)] / max_eig
            denom = s_i * (1.0 + self._lam * c_deg * c_eig)
            retention = math.exp(-delta_days / max(denom, 1e-6))
            new_tier = self._score_to_tier(retention)
            old_tier = f.tier
            f.score = retention
            if new_tier != old_tier:
                f.tier = new_tier
                tier_changes += 1
            await self.c.facts.update(f)
            updated += 1

        logger.info("forgetting cycle: updated=%d tier_changes=%d", updated, tier_changes)
        if debug is not None:
            debug.record("forgetting", {
                "updated": updated,
                "tier_changes": tier_changes,
                "total_facts": len(facts),
                "reference_time": (reference_time or now).isoformat(),
            })
        return {"updated": updated, "tier_changes": tier_changes}

    def _score_to_tier(self, score: float) -> Tier:
        if score >= self._thresholds["L0_L1"]:
            return Tier.L0
        elif score >= self._thresholds["L1_L2"]:
            return Tier.L1
        elif score >= self._thresholds["L2_L3"]:
            return Tier.L2
        else:
            return Tier.L3

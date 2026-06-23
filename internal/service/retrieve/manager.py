"""
MAS 计算中心 — 对召回结果计算激活分数。

职责：
  对 RecallService 召回的 Facts 用 MASManager 计算 MAS 分数并按分数降序返回。
"""

from __future__ import annotations

import logging
from datetime import datetime

from internal.config.settings import settings
from internal.domain.model.fact import Fact
from internal.domain.services.mas_manager import MASManager
from internal.infra.container import Container
from internal.util.call_trace import get_trace
from internal.util.debug_collector import DebugCollector

logger = logging.getLogger(__name__)


class MASComputeService:
    """对召回 Facts 执行 MAS 评分。"""

    def __init__(self, container: Container) -> None:
        self.c = container
        self._mas_manager = MASManager()

    async def compute_mas_scores(
        self,
        query_embedding: list[float],
        facts: list[Fact],
        *,
        reference_time: datetime | None = None,
        debug: DebugCollector | None = None,
    ) -> list[tuple[Fact, float]]:
        """对所有 Fact 计算 MAS 分，按分数降序返回。

        对每个 Fact：
          - edge_weight: 以该 Fact 连接的最大边权重（归一化到 [0,1]）
          - membox_event_time: 从 metadata.HappendTime 获取
          - reference_time: 透传给 recency 计算，避免墙钟差异
        """
        if (t := get_trace()) is not None:
            t.mas_calls += 1
            t.mark("mas")
        scored: list[tuple[Fact, float]] = []
        breakdowns: list[dict] = [] if debug is not None else []
        for f in facts:
            # 计算该 Fact 的 max edge weight
            edge_weight = 0.0
            try:
                edges = await self.c.edges.find_top_neighbors(f.id, top_n=3)
                if edges:
                    edge_weight = max(e.weight for e in edges)
            except Exception:
                pass

            # 获取 MemBox event time
            membox_time = f.metadata.HappendTime if f.metadata else None

            if debug is not None:
                mas, parts = self._mas_manager.compute_fact_mas(
                    query_embedding=query_embedding,
                    fact=f,
                    membox_event_time=membox_time,
                    edge_weight=edge_weight,
                    reference_time=reference_time,
                    return_breakdown=True,
                )
                scored.append((f, mas))
                breakdowns.append({
                    "id": str(f.id),
                    "content": (f.content or "")[:80],
                    "mas": float(mas),
                    "tier": f.tier.value if hasattr(f.tier, "value") else str(f.tier),
                    "happened_at": membox_time.isoformat() if membox_time else None,
                    **parts,
                })
            else:
                mas = self._mas_manager.compute_fact_mas(
                    query_embedding=query_embedding,
                    fact=f,
                    membox_event_time=membox_time,
                    edge_weight=edge_weight,
                    reference_time=reference_time,
                )
                scored.append((f, mas))

        scored.sort(key=lambda x: x[1], reverse=True)

        if debug is not None:
            breakdowns.sort(key=lambda d: d["mas"], reverse=True)
            debug.record("mas_scored", {
                "count": len(scored),
                "weights": settings.mas_weights,
                "facts": breakdowns[:60],
            })
        return scored
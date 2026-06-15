"""
记忆召回模块 — 三阶段召回 + 图随机游走。

实现 WorkFlow 设计的三种召回策略：
  1. 时间召回 — SQLite 时间范围查询
  2. 关键字召回 — LLM 提取关键词 → Entity hybrid_recall → Tag 过滤 + 向量匹配 → RRF 合并
  3. 图随机游走 — 基于转移概率公式的邻域扩展

转移概率公式：
  P(u|v,q,t) = 1/Z_v * Gamma(v,u,t) * exp(
      lambda_sem * cos(e_q, e_u)
      + lambda_mem * R(u,t)
      + lambda_struct * [ln(omega(v,u)) - alpha * ln(deg(u))]
  )
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

import numpy as np

from internal.config.settings import settings
from internal.domain.model.edge import Edge
from internal.domain.model.fact import Fact
from internal.infra.container import Container
from internal.util.debug_collector import DebugCollector
from internal.util.rrf import rrf_merge

logger = logging.getLogger(__name__)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """min-max 归一化到 [0,1]。空 dict 或所有值相等时返回原 dict（视为零信号）。"""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        # 所有值相等：给一个统一中等分数，避免该路被 min-max 抹平为 0
        return {k: 0.5 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}

# 边类型的重要性权重（用于结构项中的 iota(type)）
_TYPE_IMPORTANCE: dict[str, float] = {
    "MENTIONS": 0.7,
    "WORKS_AT": 0.9,
    "LOCATED_IN": 0.6,
    "PART_OF": 0.8,
    "CAUSED": 1.0,
    "CONTRADICTS": 0.5,
    "PREFERS": 0.8,
    "DERIVED_FROM": 0.9,
    "RELATED_TO": 0.4,
}


@dataclass
class RecallResult:
    facts: list[Fact] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)
    seed_facts: list[Fact] = field(default_factory=list)
    expanded_facts: list[Fact] = field(default_factory=list)

    def merge(self, other: RecallResult) -> None:
        seen = {str(f.id) for f in self.facts}
        for f in other.facts:
            if str(f.id) not in seen:
                self.facts.append(f)
                seen.add(str(f.id))
        self.source_scores.update(other.source_scores)
        self.expanded_facts.extend(other.expanded_facts)


class RecallService:
    """三阶段召回 + 图随机游走的编排器。"""

    def __init__(self, container: Container) -> None:
        self.c = container

    async def recall(
        self,
        query: str,
        query_embedding: list[float],
        time_range: tuple[str | None, str | None] = (None, None),
        *,
        debug: DebugCollector | None = None,
    ) -> RecallResult:
        """执行完整召回管线。

        1. 时间召回
        2. 关键字召回（Entity hybrid_recall → Fact tag 过滤 + 向量匹配 → RRF 合并）
        3. 图随机游走（从种子 Fact 出发沿着边扩展）
        返回合并去重后的结果。
        """
        result = RecallResult()

        # 1) 时间召回
        if time_range[0] or time_range[1]:
            time_facts = await self._time_recall(time_range[0], time_range[1])
            for f in time_facts:
                result.facts.append(f)
                result.source_scores[str(f.id)] = 0.5  # 中性默认分

        # 2) 关键字召回
        keyword_result = await self._keyword_recall(query, query_embedding, debug=debug)
        result.merge(keyword_result)

        # 3) 图随机游走（以种子 Fact 为起点）
        if result.facts:
            walk_result = await self._graph_walk(result.facts, query_embedding, debug=debug)
            for f in walk_result:
                if str(f.id) not in result.source_scores:
                    result.facts.append(f)
                    result.source_scores[str(f.id)] = 0.1
                    result.expanded_facts.append(f)

        # 4) 知识更新消歧：遇到 CONTRADICTS 边时只保留较新的事实
        result = await self._resolve_contradictions(result, debug=debug)

        return result

    # ---------- 时间召回 ----------

    async def _time_recall(
        self, start: str | None, end: str | None
    ) -> list[Fact]:
        return await self.c.facts.find_by_time_range(start, end, limit=100)

    # ---------- 关键字召回 ----------

    async def _keyword_recall(
        self, query: str, query_embedding: list[float],
        *, debug: DebugCollector | None = None,
    ) -> RecallResult:
        # 1) LLM 提取关键词
        fallback_used = False
        try:
            keywords = await self.c.llm.extract_entities(query)
        except Exception as e:
            logger.warning("extract_entities failed: %s", e)
            keywords = []

        if not keywords:
            keywords = [query[:20]]
            fallback_used = True

        if debug is not None:
            debug.record("keywords", {
                "raw": list(keywords),
                "fallback_used": fallback_used,
            })

        # 2) Entity hybrid_recall（Trigram + Embedding + RRF）
        top_k_entity = settings.retrieve_top_k_entity
        entities = await self.c.entities.hybrid_recall(
            keywords, query_embedding, top_k=top_k_entity
        )

        if debug is not None:
            debug.record("entity_recall", {
                "top_k": top_k_entity,
                "count": len(entities),
                "entities": [
                    {"id": str(e.id), "name": e.name, "centrality": e.centrality}
                    for e in entities
                ],
            })

        # 3) Fact 4 路召回：BM25 + Trigram + Vec + Tag
        entity_ids = [str(e.id) for e in entities]

        tag_facts: list[tuple[Fact, float]] = []
        if entity_ids:
            tag_facts = await self.c.facts.find_by_entity_ids(
                entity_ids, limit=settings.retrieve_top_k_fact_tag
            )
        vec_facts = await self.c.facts.vector_search_with_scores(
            query_embedding, top_k=settings.retrieve_top_k_fact_vec
        )
        bm25_facts = await self.c.facts.fts_search_bm25(
            keywords, top_k=settings.retrieve_top_k_fact_bm25
        )
        tri_facts = await self.c.facts.fts_search_trigram(
            keywords, top_k=settings.retrieve_top_k_fact_trigram
        )

        # 收集 id → Fact，及每路的原始分数
        id_to_fact: dict[str, Fact] = {}
        bm25_raw: dict[str, float] = {}
        vec_raw: dict[str, float] = {}
        tag_raw: dict[str, float] = {}
        tri_raw: dict[str, float] = {}
        for f, s in bm25_facts:
            fid = str(f.id)
            id_to_fact[fid] = f
            bm25_raw[fid] = s
        for f, s in vec_facts:
            fid = str(f.id)
            id_to_fact[fid] = f
            vec_raw[fid] = s
        for f, s in tag_facts:
            fid = str(f.id)
            id_to_fact[fid] = f
            tag_raw[fid] = s
        for f, s in tri_facts:
            fid = str(f.id)
            id_to_fact[fid] = f
            tri_raw[fid] = s

        if debug is not None:
            def _path_dump(facts_with_scores: list[tuple[Fact, float]]) -> list[dict]:
                return [
                    {
                        "id": str(f.id),
                        "content": (f.content or "")[:80],
                        "raw_score": float(s),
                        "happened_at": f.metadata.HappendTime.isoformat() if (f.metadata and f.metadata.HappendTime) else None,
                    }
                    for f, s in facts_with_scores
                ]
            debug.record("fact_recall_per_path", {
                "lambdas": {
                    "bm25": settings.retrieve_lambda_bm25,
                    "vec": settings.retrieve_lambda_vec,
                    "tag": settings.retrieve_lambda_tag,
                    "trigram": settings.retrieve_lambda_trigram,
                },
                "top_k": {
                    "bm25": settings.retrieve_top_k_fact_bm25,
                    "vec": settings.retrieve_top_k_fact_vec,
                    "tag": settings.retrieve_top_k_fact_tag,
                    "trigram": settings.retrieve_top_k_fact_trigram,
                },
                "bm25": _path_dump(bm25_facts),
                "vec": _path_dump(vec_facts),
                "tag": _path_dump(tag_facts),
                "trigram": _path_dump(tri_facts),
            })

        # 各路独立归一化（min-max 到 [0,1]）
        bm25_n = _normalize(bm25_raw)
        vec_n = _normalize(vec_raw)
        tag_n = _normalize(tag_raw)
        tri_n = _normalize(tri_raw)

        # 加权求和
        fused: list[tuple[str, float]] = []
        for fid in id_to_fact:
            score = (
                settings.retrieve_lambda_bm25 * bm25_n.get(fid, 0.0)
                + settings.retrieve_lambda_vec * vec_n.get(fid, 0.0)
                + settings.retrieve_lambda_tag * tag_n.get(fid, 0.0)
                + settings.retrieve_lambda_trigram * tri_n.get(fid, 0.0)
            )
            fused.append((fid, score))
        fused.sort(key=lambda kv: kv[1], reverse=True)

        result = RecallResult()
        for fid, score in fused:
            fact = id_to_fact.get(fid)
            if fact:
                result.facts.append(fact)
                result.source_scores[fid] = score
        result.seed_facts = list(result.facts)

        if debug is not None:
            debug.record("fact_recall_fused", {
                "total": len(fused),
                "top": [
                    {
                        "id": fid,
                        "content": (id_to_fact[fid].content or "")[:80],
                        "fused": float(score),
                        "bm25_n": float(bm25_n.get(fid, 0.0)),
                        "vec_n": float(vec_n.get(fid, 0.0)),
                        "tag_n": float(tag_n.get(fid, 0.0)),
                        "tri_n": float(tri_n.get(fid, 0.0)),
                        "happened_at": id_to_fact[fid].metadata.HappendTime.isoformat() if (id_to_fact[fid].metadata and id_to_fact[fid].metadata.HappendTime) else None,
                    }
                    for fid, score in fused[:50]
                ],
            })

        return result

    # ---------- 图随机游走 ----------

    async def _graph_walk(
        self,
        seed_facts: list[Fact],
        query_embedding: list[float],
        *,
        debug: DebugCollector | None = None,
    ) -> list[Fact]:
        """从种子 Fact 出发，按转移概率公式做有界随机游走。"""
        now = datetime.now(timezone.utc)
        max_depth = settings.retrieve_max_graph_depth
        threshold = settings.retrieve_graph_walk_threshold

        visited: set[str] = {str(f.id) for f in seed_facts}
        frontier: list[Fact] = list(seed_facts)
        expanded: list[Fact] = []

        for _depth in range(max_depth):
            next_frontier: list[Fact] = []
            for fact in frontier:
                fid = str(fact.id)
                deg_v = await self.c.edges.degree(UUID(fid))
                if deg_v == 0:
                    continue
                neighbors = await self.c.edges.find_top_neighbors(
                    UUID(fid), top_n=5
                )
                if not neighbors:
                    continue

                # 计算 Z_v（邻居的未归一化分子之和）
                neighbor_data: list[tuple[Edge, Fact, float]] = []
                unnorm_sum = 0.0
                for edge in neighbors:
                    # 确定邻居 Fact 的 id
                    nid = (str(edge.to_fact_id)
                           if str(edge.from_fact_id) == fid
                           else str(edge.from_fact_id))
                    if nid in visited:
                        continue
                    nbr = await self._load_fact_from_edge(nid, edge)
                    if nbr is None or nbr.embedding is None:
                        continue
                    deg_u = await self.c.edges.degree(UUID(nid))
                    p_raw = self._transition_probability(
                        query_embedding, fact, nbr, edge, deg_u, now
                    )
                    if p_raw > 0:
                        neighbor_data.append((edge, nbr, p_raw))
                        unnorm_sum += p_raw

                if unnorm_sum <= 0:
                    continue

                for edge, nbr, p_raw in neighbor_data:
                    prob = p_raw / unnorm_sum  # 归一化
                    if prob < threshold:
                        continue
                    visited.add(str(nbr.id))
                    expanded.append(nbr)
                    next_frontier.append(nbr)

            if not next_frontier:
                break
            frontier = next_frontier

        if debug is not None:
            debug.record("graph_walk", {
                "seed_count": len(seed_facts),
                "expanded_count": len(expanded),
                "max_depth": max_depth,
                "threshold": threshold,
                "expanded": [
                    {"id": str(f.id), "content": (f.content or "")[:80]}
                    for f in expanded[:30]
                ],
            })

        return expanded

    async def _load_fact_from_edge(
        self, fact_id: str, edge: Edge
    ) -> Fact | None:
        """从边的一侧加载 Fact 对象。"""
        try:
            return await self.c.facts.get_by_id(UUID(fact_id))
        except Exception:
            return None

    async def _resolve_contradictions(
        self, result: RecallResult,
        *, debug: DebugCollector | None = None,
    ) -> RecallResult:
        """知识更新消歧：检测 CONTRADICTS 边，只保留较新的事实。

        当两个被召回的事实之间存在 CONTRADICTS 边时，说明发生了知识更新。
        旧事实应当被抑制，保留新事实。
        """
        if not result.facts:
            return result

        fact_map: dict[str, Fact] = {str(f.id): f for f in result.facts}

        # 收集所有 CONTRADICTS 边
        stale_ids: set[str] = set()
        for f in result.facts:
            fid = str(f.id)
            edges = await self.c.edges.find_by_fact(UUID(fid))
            for e in edges:
                if str(e.info) != "CONTRADICTS":
                    continue
                # 确定对方 fact id
                other_id = (str(e.to_fact_id)
                            if str(e.from_fact_id) == fid
                            else str(e.from_fact_id))
                if other_id not in fact_map:
                    continue
                # 比较创建时间，保留较新的
                    f_other = fact_map[other_id]
                    t_fid = f.created_at
                    t_other = f_other.created_at
                    if t_fid and t_other:
                        if t_fid < t_other:
                            stale_ids.add(fid)
                        else:
                            stale_ids.add(other_id)

        if not stale_ids:
            if debug is not None:
                debug.record("contradiction", {
                    "dropped_count": 0, "kept_count": len(result.facts), "dropped": [],
                })
            return result

        # 过滤掉旧事实
        kept = [f for f in result.facts if str(f.id) not in stale_ids]
        logger.info("Contradiction resolution: dropped %d stale facts, kept %d",
                     len(stale_ids), len(kept))
        if debug is not None:
            dropped_facts = [f for f in result.facts if str(f.id) in stale_ids]
            debug.record("contradiction", {
                "dropped_count": len(stale_ids),
                "kept_count": len(kept),
                "dropped": [
                    {"id": str(f.id), "content": (f.content or "")[:80]}
                    for f in dropped_facts
                ],
            })
        result.facts = kept
        return result

    def _transition_probability(
        self,
        query_embedding: list[float],
        fact_v: Fact,
        fact_u: Fact,
        edge: Edge,
        deg_u: int,
        current_time: datetime,
    ) -> float:
        """计算转移概率 P(u|v,q,t)。

        公式：
          Gamma(v,u,t) * exp(
              lambda_sem * cos(e_q, e_u)
              + lambda_mem * R(u,t)
              + lambda_struct * [ln(omega(v,u)) - alpha * ln(deg(u))]
          )
        """
        # ---- 时间硬门控 Gamma(v,u,t) ----
        t_valid = getattr(edge, "t_valid", None)
        t_invalid = getattr(edge, "t_invalid", None)
        t_str = current_time.isoformat()
        if t_valid and t_str < t_valid:
            return 0.0
        if t_invalid and t_str > t_invalid:
            return 0.0

        # ---- 语义项 cos(e_q, e_u) ----
        if fact_u.embedding is not None:
            q_vec = np.array(query_embedding)
            u_vec = np.array(fact_u.embedding)
            norm = np.linalg.norm(q_vec) * np.linalg.norm(u_vec)
            cos_sim = float(np.dot(q_vec, u_vec) / norm) if norm > 0 else 0.0
        else:
            cos_sim = 0.0
        sem_term = settings.retrieve_lambda_sem * cos_sim

        # ---- 记忆项 R(u,t) ----
        ref_time = fact_u.last_accessed_at or fact_u.created_at
        if ref_time:
            delta_days = (current_time - ref_time).total_seconds() / 86400.0
        else:
            delta_days = 0.0
        tau = settings.retrieve_tau_days
        eta = settings.retrieve_eta
        n_u = fact_u.access_count
        r_val = math.exp(-delta_days / (tau * (1.0 + eta * math.log(1.0 + n_u))))
        mem_term = settings.retrieve_lambda_mem * r_val

        # ---- 结构项 ln(omega(v,u)) - alpha * ln(deg(u)) ----
        confidence = getattr(edge, "confidence", 1.0)
        edge_info = str(edge.info) if not isinstance(edge.info, str) else edge.info
        type_imp = _TYPE_IMPORTANCE.get(edge_info, 0.4)
        omega = confidence * type_imp  # (0, 1]
        omega = max(omega, 1e-6)
        struct_term = settings.retrieve_lambda_struct * (
            math.log(omega) - settings.retrieve_alpha * math.log(max(deg_u, 1))
        )

        # ---- 指数合并 ----
        return math.exp(sem_term + mem_term + struct_term)
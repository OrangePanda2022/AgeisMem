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
from internal.util.call_trace import get_trace
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

        # 2) 关键字召回（或投机召回）
        if settings.speculative_retrieval_enabled:
            # [方案B] Draft-then-Verify：先跑向量一路，检查充分性
            draft_result = await self._draft_recall(query_embedding, debug=debug)
            if draft_result.facts:
                draft_summary_parts = [
                    f"- {f.content[:100]}" for f in draft_result.facts[:5]
                ]
                draft_summary = "\n".join(draft_summary_parts)
                try:
                    suff = await self.c.llm.check_sufficiency(query, draft_summary)
                    if (suff.get("sufficient", False)
                            and suff.get("confidence", 0) >= settings.speculative_confidence_threshold):
                        logger.info("Speculative retrieval: draft sufficient (conf=%.2f)",
                                     suff.get("confidence", 0))
                        result.merge(draft_result)
                        # 跳过完整 4-path，直接进图游走
                        if result.facts:
                            walk_result = await self._graph_walk(result.facts, query_embedding, debug=debug)
                            for f in walk_result:
                                if str(f.id) not in result.source_scores:
                                    result.facts.append(f)
                                    result.source_scores[str(f.id)] = 0.1
                                    result.expanded_facts.append(f)
                        result = await self._resolve_contradictions(result, debug=debug)
                        if debug is not None:
                            debug.record("speculative_recall", {
                                "draft_sufficient": True,
                                "confidence": suff.get("confidence", 0),
                            })
                        return result
                except Exception as e:
                    logger.warning("Speculative sufficiency check failed: %s", e)

            # Draft 不充分，回退到完整 4-path
            keyword_result = await self._keyword_recall(query, query_embedding, debug=debug)
            result.merge(keyword_result)
        else:
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
        *, pre_extracted_entities: list[str] | None = None,
        debug: DebugCollector | None = None,
    ) -> RecallResult:
        # 单题调用追踪：4 路召回触发计数（主召回/反向扩展/迭代轮均经此唯一入口）
        if (t := get_trace()) is not None:
            t.recall_calls += 1
            t.mark("keyword_recall")
        # 1) LLM 提取关键词（或使用预提取的关键词）
        fallback_used = False
        if pre_extracted_entities is not None:
            keywords = list(pre_extracted_entities)
        else:
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

        # P1: KB-aware 关键词映射 — 用 KB 中实际存在的实体替换原始关键词
        # 避免搜索 KB 中不存在的实体词（如 "5-day trip" vs KB 中的 "Costa Rica"）
        mapped_keywords: list[str] = []
        seen_mapped: set[str] = set()
        if keywords:
            try:
                kb_entities = await self.c.entities.hybrid_recall(
                    keywords, query_embedding, top_k=10
                )
                kb_names = {e.name for e in kb_entities}
                # 优先用 KB 实体名替换；对原始关键词，如果 KB 中有匹配则保留原词，否则丢弃
                for kw in keywords:
                    kw_lower = kw.lower()
                    # 关键词在 KB 中有近似匹配 → 保留
                    matched = any(kw_lower in n.lower() or n.lower() in kw_lower for n in kb_names)
                    if matched:
                        if kw_lower not in seen_mapped:
                            mapped_keywords.append(kw)
                            seen_mapped.add(kw_lower)
                    # 否则用 top KB 实体补充
                for e in kb_entities[:5]:
                    n_lower = e.name.lower()
                    if n_lower not in seen_mapped:
                        mapped_keywords.append(e.name)
                        seen_mapped.add(n_lower)
                if mapped_keywords:
                    logger.info("KB keyword mapping: %s → %s", keywords[:8], mapped_keywords[:8])
                    keywords = mapped_keywords
            except Exception as e:
                logger.warning("KB keyword mapping failed, using raw keywords: %s", e)

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

        # 单题调用追踪:每路找回 fact 数 + RRF 融合去重后数(累计,跨多次召回)
        if (t := get_trace()) is not None:
            t.add_recall_facts(
                bm25=len(bm25_facts), trigram=len(tri_facts),
                vec=len(vec_facts), tag=len(tag_facts),
                fused=len(fused),
            )

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

    # ---------- 迭代召回（方案A：查询扩展） ----------

    async def recall_by_keywords(
        self,
        keywords: list[str],
        query_embedding: list[float],
        existing_fact_ids: set[str] | None = None,
        *,
        debug: DebugCollector | None = None,
    ) -> RecallResult:
        """用扩展关键词执行定向召回（用于迭代检索第二轮）。

        跳过 LLM 实体抽取（关键词即实体），跑 _keyword_recall + 图游走。
        过滤掉已召回的事实，仅返回新增事实。

        Args:
            keywords: 充分性检查生成的替代搜索关键词。
            query_embedding: 原始查询向量（用于向量搜索 + 图游走）。
            existing_fact_ids: 已召回事实 ID 集合（用于去重）。
        """
        keyword_result = await self._keyword_recall(
            "iterative_query", query_embedding,
            pre_extracted_entities=keywords, debug=debug,
        )

        # 过滤已存在的 facts
        if existing_fact_ids:
            keyword_result.facts = [
                f for f in keyword_result.facts
                if str(f.id) not in existing_fact_ids
            ]
            keyword_result.source_scores = {
                k: v for k, v in keyword_result.source_scores.items()
                if k not in existing_fact_ids
            }

        # 图游走扩展新发现的 seed facts
        if keyword_result.facts:
            walk_result = await self._graph_walk(
                keyword_result.facts, query_embedding, debug=debug,
            )
            existing = existing_fact_ids or set()
            for f in walk_result:
                fid = str(f.id)
                if fid not in existing and fid not in keyword_result.source_scores:
                    keyword_result.facts.append(f)
                    keyword_result.source_scores[fid] = 0.1
                    keyword_result.expanded_facts.append(f)

        if debug is not None:
            debug.record("iterative_recall", {
                "keywords": keywords,
                "new_facts_count": len(keyword_result.facts),
                "expanded_count": len(keyword_result.expanded_facts),
            })

        return keyword_result

    async def expand_keywords_via_graph(
        self,
        seed_facts: list[Fact],
        *,
        top_n_facts: int = 5,
        top_n_neighbors_per_fact: int = 3,
        debug: DebugCollector | None = None,
    ) -> list[str]:
        """从 seed facts 出发，沿 edges 找邻居 fact，提取其 tag 实体名作为扩展关键词。

        用于 alt_keywords 补充：LLM 推断的 alt_keywords 常是抽象主题词，
        无法命中用户实际提过的具体品牌/型号；从 graph 邻居 fact 的 tag
        实体中取具体名词，可补足这部分。
        """
        if not seed_facts:
            return []

        seen_entity_names: set[str] = set()
        neighbor_entity_names: list[str] = []

        for fact in seed_facts[:top_n_facts]:
            try:
                neighbors = await self.c.edges.find_top_neighbors(
                    fact.id, top_n=top_n_neighbors_per_fact,
                )
            except Exception as e:
                logger.warning("expand_keywords_via_graph: find_top_neighbors failed for %s: %s", fact.id, e)
                continue
            for edge in neighbors:
                nid = (str(edge.to_fact_id)
                       if str(edge.from_fact_id) == str(fact.id)
                       else str(edge.from_fact_id))
                try:
                    nbr = await self.c.facts.get_by_id(UUID(nid))
                except Exception:
                    nbr = None
                if nbr is None:
                    continue
                # 从 neighbor fact 的 tag 提取实体名
                for tag in nbr.tag[:3]:
                    name = (tag.Entity.name or "").strip()
                    if not name:
                        continue
                    key = name.lower()
                    if key in seen_entity_names:
                        continue
                    seen_entity_names.add(key)
                    neighbor_entity_names.append(name)

        if debug is not None:
            debug.record("expand_keywords_via_graph", {
                "seed_count": len(seed_facts[:top_n_facts]),
                "expanded_keywords": neighbor_entity_names,
                "count": len(neighbor_entity_names),
            })

        return neighbor_entity_names

    # ---------- 投机召回（方案B：Draft-then-Verify） ----------

    async def _draft_recall(
        self, query_embedding: list[float],
        *, debug: DebugCollector | None = None,
    ) -> RecallResult:
        """仅跑向量搜索一路作为 draft，跳过实体抽取和其他三路。"""
        if (t := get_trace()) is not None:
            t.draft_calls += 1
            t.mark("draft_recall")
        vec_facts = await self.c.facts.vector_search_with_scores(
            query_embedding, top_k=settings.retrieve_top_k_fact_vec,
        )

        vec_raw: dict[str, float] = {}
        id_to_fact: dict[str, Fact] = {}
        for f, s in vec_facts:
            fid = str(f.id)
            id_to_fact[fid] = f
            vec_raw[fid] = s

        vec_n = _normalize(vec_raw)

        result = RecallResult()
        for fid, score_n in vec_n.items():
            fact = id_to_fact.get(fid)
            if fact:
                result.facts.append(fact)
                result.source_scores[fid] = score_n
        result.seed_facts = list(result.facts)

        # 单题调用追踪:投机召回仅 vec 一路,fused 即去重后 result.facts(累计)
        if (t := get_trace()) is not None:
            t.add_recall_facts(vec=len(vec_facts), fused=len(result.facts))

        if debug is not None:
            debug.record("draft_recall", {
                "facts_count": len(result.facts),
                "top": [
                    {"id": fid, "content": (id_to_fact[fid].content or "")[:80], "vec_n": float(score_n)}
                    for fid, score_n in sorted(vec_n.items(), key=lambda kv: kv[1], reverse=True)[:10]
                ],
            })

        return result

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

        # 单题调用追踪：图游走节点数累计（一题可能多次 graph_walk，累加而非覆盖）
        if (t := get_trace()) is not None:
            t.graph_walk_calls += 1
            t.graph_walk_nodes_visited += len(visited)   # 含种子节点
            t.graph_walk_nodes_expanded += len(expanded)  # 仅新增节点
            t.mark("graph_walk")

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
        if ref_time and ref_time.tzinfo is None:
            # DB 中存的 naive datetime 视为 UTC
            ref_time = ref_time.replace(tzinfo=timezone.utc)
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
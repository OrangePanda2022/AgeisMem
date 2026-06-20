"""写入管线：fact 抽取 → Tag 填充 → 邻域召回 → 子图 → LLM 决策 → 落库。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from internal.domain.model.edge import Edge
from internal.domain.model.entity import Entity
from internal.domain.model.fact import Fact
from internal.domain.model.membox import MemBox
from internal.domain.model.metadata import Metadata
from internal.domain.model.tag import Tag
from internal.domain.model.tier import Tier
from internal.infra.container import Container

logger = logging.getLogger(__name__)


def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _meta_from_dict(d: dict) -> Metadata:
    return Metadata(
        Person=d.get("Person", "") or "",
        Object=d.get("Object", "") or "",
        Location=d.get("Location", "") or "",
        Event=d.get("Event", "") or "",
        Organization=d.get("Organization", "") or "",
        Preference=d.get("Preference", "") or "",
        HappendTime=_parse_dt(d.get("HappendTime")),
        MentionedTime=_parse_dt(d.get("MentionedTime")),
        History=d.get("History"),
    )


class WriteService:
    """orchestrate 记忆写入流程。"""

    def __init__(self, container: Container) -> None:
        self.c = container

    async def ingest(
        self,
        raw_message: str,
        event_time: datetime | None = None,
        provenance: dict | None = None,
    ) -> list[Fact]:
        """处理一条对话消息：抽取 N 个 fact，逐个写入图。返回写入的 facts。

        provenance（可选）携带消息来源信息（source_session_id/source_turn_index/
        source_turn_role），写入到每个新 fact 的 metadata，用于评测场景回溯。
        """
        ts_str = event_time.date().isoformat() if event_time else ""
        try:
            extraction = await self.c.llm.extract_facts(raw_message, event_time=ts_str)
        except Exception as e:
            logger.warning("extract_facts failed: %s", e)
            return []

        # LLM 有时返回 list（直接是 facts 数组）而非 {"facts": [...]}，做兼容
        facts_list: list[dict] = []
        if isinstance(extraction, list):
            facts_list = extraction
        elif isinstance(extraction, dict):
            facts_list = extraction.get("facts", [])
        else:
            logger.warning("Unexpected extraction type: %s", type(extraction))
            return []

        results: list[Fact] = []
        for item in facts_list:
            try:
                pre_entities = item.get("entities", [])
                fact = await self._process_fact(item, raw_message, event_time, pre_entities, provenance=provenance)
                if fact:
                    results.append(fact)
            except Exception as e:
                logger.warning("process_fact failed: %s", e, exc_info=True)
        return results

    async def _process_fact(
        self, item: dict, raw_message: str, event_time: datetime | None,
        pre_extracted_entities: list[str] | None = None,
        *,
        provenance: dict | None = None,
    ) -> Fact | None:
        content = (item.get("content") or "").strip()
        if not content:
            return None
        meta = _meta_from_dict(item.get("metadata", {}))
        if provenance:
            meta.source_session_id = provenance.get("source_session_id", "")
            meta.source_turn_index = provenance.get("source_turn_index")
            meta.source_turn_role = provenance.get("source_turn_role", "")
        if event_time and meta.MentionedTime is None:
            meta.MentionedTime = event_time

        # P0: 回填 HappendTime — 当 HappendTime 为空但有 MentionedTime 时，
        # 将 MentionedTime 近似作为 HappendTime（事件时间信号的兜底）
        if meta.HappendTime is None and meta.MentionedTime is not None:
            meta.HappendTime = meta.MentionedTime

        # 1) Fact embedding
        embedding = await self.c.embedder.embed_text(content)

        # 2) Tag 填充：用 fact 提取时已获得的 entities 跳过 LLM 调用
        tags = await self._build_tags(content, pre_extracted_entities, fact_embedding=embedding)

        # 3) 三层 Fact 召回：vector ∪ tag-filter ∪ recency
        candidates = await self._three_stage_recall(content, embedding, tags)

        # 4) 子图加载：每个候选取 top-3 边邻 + Box summary
        subgraph = await self._load_subgraph(candidates)

        # 5) LLM 演化决策
        decision = await self._evolution_decision(content, subgraph)

        # 6) 应用决策
        new_fact = Fact(
            id=uuid4(),
            content=content,
            original_msg=item.get("original_msg") or raw_message,
            embedding=embedding,
            metadata=meta,
            tag=tags,
            tier=Tier.L0,
            score=0.5,
            created_at=event_time or datetime.now(timezone.utc),
        )

        # MemBox 默认装载（最近一个 box；没有就新建以 fact content 头10字为 title）
        membox = await self._pick_or_create_membox(content)
        new_fact.membox_id = membox.id

        action = decision.get("decision", "ADD")
        if action == "NOOP":
            logger.info("Decision NOOP: skipping fact '%s'", content[:30])
            return None
        if action == "MERGE":
            target_id = decision.get("target_fact_id")
            if target_id:
                try:
                    target = await self.c.facts.get_by_id(UUID(target_id))
                    if target:
                        target.bump_score(0.05)
                        # 合并 tag
                        existing_eids = {str(t.Entity.id) for t in target.tag}
                        for t in tags:
                            if str(t.Entity.id) not in existing_eids:
                                target.tag.append(t)
                        await self.c.facts.update(target)
                        return target
                except Exception:
                    pass

        # ADD / LINK / UPDATE 均要写入新 Fact
        await self.c.facts.add(new_fact)

        # 处理 edge_changes
        for ec in decision.get("edge_changes", []) or []:
            await self._apply_edge_change(new_fact.id, ec, conflict=decision.get("conflict", False))

        return new_fact

    # ---------- 内部组件 ----------

    async def _build_tags(self, content: str, pre_extracted_entities: list[str] | None = None, *, fact_embedding: list[float] | None = None) -> list[Tag]:
        if pre_extracted_entities:
            keywords = pre_extracted_entities
        else:
            keywords = await self.c.llm.extract_entities(content)
        if not keywords:
            return []
        # 对每个 keyword 做 hybrid recall；候选并集后让 LLM 取舍是另一次 LLM 调用，
        # 这里为节流采用启发式：直接复用 hybrid 召回 top-3 作为已有，
        # 未命中的关键词作为新 Entity 创建，全部权重默认 1.0（后续 evolution 可调整）。
        query_emb = fact_embedding if fact_embedding is not None else await self.c.embedder.embed_text(content)
        recalled = await self.c.entities.hybrid_recall(keywords, query_emb, top_k=10)
        recalled_names = {e.name for e in recalled}
        tags: list[Tag] = []
        # 先收集需新建的 keyword，批量嵌入
        new_keywords = [kw for kw in keywords if kw not in recalled_names]
        new_embs: dict[str, list[float]] = {}
        if new_keywords:
            batch = await self.c.embedder.embed_texts(new_keywords)
            for kw, emb in zip(new_keywords, batch):
                new_embs[kw] = emb
        for kw in keywords:
            existing = next((e for e in recalled if e.name == kw), None)
            if existing is None and kw in recalled_names:
                existing = next(e for e in recalled if e.name == kw)
            if existing is None:
                # 新建，用预计算的批量 embedding
                ent = Entity(name=kw, embedding=new_embs[kw])
                await self.c.entities.add(ent)
                tags.append(Tag(Entity=ent, Weight=1.0))
            else:
                tags.append(Tag(Entity=existing, Weight=1.0))
        # 把召回到但 keywords 没明确出现的 top entities 也带上（弱权重），增强连接
        for ent in recalled[:3]:
            if ent.name not in keywords:
                tags.append(Tag(Entity=ent, Weight=0.3))
        return tags

    async def _three_stage_recall(
        self, content: str, embedding: list[float], tags: list[Tag]
    ) -> list[Fact]:
        # 1) 向量
        vec_hits = await self.c.facts.vector_search(embedding, top_k=10)
        # 2) Tag 过滤
        ent_ids = [str(t.Entity.id) for t in tags]
        tag_hits = await self.c.facts.find_by_entity_ids(ent_ids, limit=10)
        # 3) recency
        recent = await self.c.facts.find_recent(limit=10)

        seen: dict[str, Fact] = {}
        for f in vec_hits:
            seen[str(f.id)] = f
        for f, _ in tag_hits:
            seen[str(f.id)] = f
        for f in recent:
            seen[str(f.id)] = f
        return list(seen.values())

    async def _load_subgraph(self, candidates: list[Fact]) -> dict:
        sub: dict = {"facts": [], "edges": [], "memboxes": {}}
        for f in candidates[:8]:  # 限制规模
            edges = await self.c.edges.find_top_neighbors(f.id, top_n=3)
            sub["facts"].append({
                "id": str(f.id),
                "content": f.content,
                "tier": f.tier.value,
                "score": f.score,
            })
            for e in edges:
                sub["edges"].append({
                    "id": str(e.id),
                    "from": str(e.from_fact_id),
                    "to": str(e.to_fact_id),
                    "info": e.info if isinstance(e.info, str) else "RELATED_TO",
                    "weight": e.weight,
                })
            if f.membox_id and str(f.membox_id) not in sub["memboxes"]:
                box = await self.c.memboxes.get_by_id(f.membox_id)
                if box:
                    sub["memboxes"][str(box.id)] = {
                        "title": box.title,
                        "summary": box.summary,
                    }
        return sub

    async def _evolution_decision(self, content: str, subgraph: dict) -> dict:
        if not subgraph["facts"]:
            return {"decision": "ADD", "edge_changes": [], "conflict": False, "reason": "图为空"}
        existing = [
            {"id": f["id"], "content": f["content"]}
            for f in subgraph["facts"]
        ]
        membox_ctx = next(iter(subgraph["memboxes"].values()), {})
        try:
            return await self.c.llm.evolution_decision(content, membox_ctx, existing)
        except Exception as e:
            logger.warning("evolution_decision failed: %s; default ADD", e)
            return {"decision": "ADD", "edge_changes": [], "conflict": False, "reason": "fallback"}

    async def _apply_edge_change(self, new_fact_id, ec: dict, conflict: bool) -> None:
        op = ec.get("op")
        target_str = ec.get("target_fact_id")
        if not target_str:
            return
        try:
            target_id = UUID(target_str)
        except Exception:
            return
        if op == "create":
            edge = Edge(
                from_fact_id=new_fact_id,
                to_fact_id=target_id,
                info=ec.get("info") or "RELATED_TO",
                weight=float(ec.get("weight", 0.5)),
            )
            setattr(edge, "confidence", float(ec.get("confidence", 1.0)))
            if conflict:
                edge.history = [f"conflict@{datetime.now(timezone.utc).isoformat()}"]
            await self.c.edges.add(edge)
        elif op == "update_weight":
            # 查找该方向是否已有边
            edges = await self.c.edges.find_by_fact(new_fact_id)
            for e in edges:
                if str(e.to_fact_id) == target_str or str(e.from_fact_id) == target_str:
                    await self.c.edges.update_weight(e.id, float(ec.get("weight", e.weight)))
                    break
        elif op == "update_type":
            # 简化：新建一条覆盖（保留旧 history）
            await self._apply_edge_change(new_fact_id, {**ec, "op": "create"}, conflict=conflict)

    async def _pick_or_create_membox(self, content: str) -> MemBox:
        # MVP 策略：每个会话进程共享一个默认 box；当不存在时创建。
        existing = await self.c.memboxes.list_all(limit=1)
        if existing:
            box = existing[0]
            box.record_access()
            await self.c.memboxes.update(box)
            return box
        box = MemBox(title="default", summary="默认记忆容器")
        await self.c.memboxes.add(box)
        return box

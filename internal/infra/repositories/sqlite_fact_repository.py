"""Fact 仓储的 SQLite 实现。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID

from internal.domain.model.fact import Fact
from internal.domain.model.metadata import Metadata
from internal.domain.model.tag import Tag
from internal.domain.model.tier import Tier
from internal.domain.repositories.fact_repository import FactRepository
from internal.infra.database.sqlite import Database, serialize_embedding

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse(s: str | None) -> datetime | None:
    if s is None or s == "":
        return None
    return datetime.fromisoformat(s)


class SQLiteFactRepository(FactRepository):
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _get_or_assign_vec_rowid(self, fact_id: str) -> int:
        row = await self._db.fetchone(
            "SELECT vec_rowid FROM fact_rowid_map WHERE fact_id = ?", (fact_id,)
        )
        if row is not None and row["vec_rowid"] is not None:
            return int(row["vec_rowid"])
        max_row = await self._db.fetchone(
            "SELECT COALESCE(MAX(vec_rowid), 0) + 1 AS n FROM fact_rowid_map"
        )
        next_id = int(max_row["n"]) if max_row else 1
        await self._db.execute(
            "INSERT OR REPLACE INTO fact_rowid_map(fact_id, vec_rowid) VALUES (?,?)",
            (fact_id, next_id),
        )
        return next_id

    async def add(self, fact: Fact) -> None:
        fid = str(fact.id)
        meta = fact.metadata
        meta_json = json.dumps(
            {
                "Person": meta.Person,
                "Object": meta.Object,
                "Location": meta.Location,
                "Event": meta.Event,
                "Organization": meta.Organization,
                "Preference": meta.Preference,
                "HappendTime": _iso(meta.HappendTime),
                "MentionedTime": _iso(meta.MentionedTime),
                "History": meta.History,
            },
            ensure_ascii=False,
        )
        await self._db.execute(
            """INSERT OR REPLACE INTO facts
               (id, membox_id, content, original_msg, score, tier,
                access_count, created_at, last_accessed_at, metadata_json, happened_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fid,
                str(fact.membox_id) if fact.membox_id else None,
                fact.content,
                fact.original_msg,
                fact.score,
                fact.tier.value,
                fact.access_count,
                _iso(fact.created_at) or _now(),
                _iso(fact.last_accessed_at),
                meta_json,
                _iso(meta.HappendTime),
            ),
        )
        # Tags
        await self._db.execute("DELETE FROM tags WHERE fact_id = ?", (fid,))
        if fact.tag:
            await self._db.executemany(
                "INSERT OR REPLACE INTO tags(fact_id, entity_id, weight) VALUES (?,?,?)",
                [(fid, str(t.Entity.id), t.Weight) for t in fact.tag],
            )
        # FTS + Vector 共用同一个 vec_rowid
        rowid = await self._get_or_assign_vec_rowid(fid)
        # 双 FTS5 索引（unicode61 跑 BM25 + trigram 兜底中文/OOV/短词）
        for table in ("facts_fts_word", "facts_fts_tri"):
            await self._db.execute(
                f"DELETE FROM {table} WHERE rowid = ?", (rowid,)
            )
            await self._db.execute(
                f"INSERT INTO {table}(rowid, content, original_msg) VALUES (?, ?, ?)",
                (rowid, fact.content or "", fact.original_msg or ""),
            )
        # Vector
        if fact.embedding is not None:
            await self._db.execute(
                "DELETE FROM facts_vec WHERE rowid = ?", (rowid,)
            )
            await self._db.execute(
                "INSERT INTO facts_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, serialize_embedding(fact.embedding)),
            )

    async def batch_add(self, facts: list[Fact]) -> None:
        for f in facts:
            await self.add(f)

    async def get_by_id(self, fact_id: UUID) -> Fact | None:
        row = await self._db.fetchone(
            "SELECT * FROM facts WHERE id = ?", (str(fact_id),)
        )
        if row is None:
            return None
        return await self._row_to_fact(row)

    async def find_by_membox_id(self, membox_id: UUID) -> list[Fact]:
        rows = await self._db.fetchall(
            "SELECT * FROM facts WHERE membox_id = ?", (str(membox_id),)
        )
        return [await self._row_to_fact(r) for r in rows]

    async def vector_search(
        self, embedding: list[float], top_k: int = 20
    ) -> list[Fact]:
        facts_with_scores = await self.vector_search_with_scores(embedding, top_k)
        return [f for f, _ in facts_with_scores]

    async def vector_search_with_scores(
        self, embedding: list[float], top_k: int = 20
    ) -> list[tuple[Fact, float]]:
        rows = await self._db.fetchall(
            """SELECT m.fact_id AS id, v.distance AS dist
               FROM facts_vec v
               JOIN fact_rowid_map m ON m.vec_rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (serialize_embedding(embedding), top_k),
        )
        results: list[tuple[Fact, float]] = []
        for r in rows:
            f = await self.get_by_id(UUID(r["id"]))
            if f:
                # L2 距离转近似相似度 (1 - normalized_dist)
                sim = 1.0 / (1.0 + float(r["dist"]))
                results.append((f, sim))
        return results

    async def fts_search_bm25(
        self, keywords: list[str], top_k: int = 20
    ) -> list[tuple[Fact, float]]:
        """BM25 召回（unicode61 分词）。content 权重 2.0、original_msg 权重 1.0。

        返回 [(Fact, score)]：score 已转为正数（FTS5 bm25() 越小越相关，
        这里取负后保持 "越大越相关" 的语义；归一化由调用方处理）。
        """
        clauses = [f'"{kw.strip()}"' for kw in keywords if len(kw.strip()) >= 2]
        if not clauses:
            return []
        fts_query = " OR ".join(clauses)
        try:
            rows = await self._db.fetchall(
                """SELECT m.fact_id AS id, bm25(facts_fts_word, 2.0, 1.0) AS s
                   FROM facts_fts_word f
                   JOIN fact_rowid_map m ON m.vec_rowid = f.rowid
                   WHERE facts_fts_word MATCH ?
                   ORDER BY s
                   LIMIT ?""",
                (fts_query, top_k),
            )
        except Exception as e:
            logger.warning("fts_search_bm25 failed (query=%r): %s", fts_query, e)
            return []
        results: list[tuple[Fact, float]] = []
        for r in rows:
            f = await self.get_by_id(UUID(r["id"]))
            if f:
                results.append((f, -float(r["s"])))
        return results

    async def fts_search_trigram(
        self, keywords: list[str], top_k: int = 20
    ) -> list[tuple[Fact, float]]:
        """Trigram 召回（中文/OOV/短词兜底）。返回 [(Fact, 1/(rank_idx+1))]。"""
        clauses = [f'"{kw.strip()}"' for kw in keywords if len(kw.strip()) >= 3]
        if not clauses:
            return []
        fts_query = " OR ".join(clauses)
        try:
            rows = await self._db.fetchall(
                """SELECT m.fact_id AS id
                   FROM facts_fts_tri f
                   JOIN fact_rowid_map m ON m.vec_rowid = f.rowid
                   WHERE facts_fts_tri MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, top_k),
            )
        except Exception as e:
            logger.warning("fts_search_trigram failed (query=%r): %s", fts_query, e)
            return []
        results: list[tuple[Fact, float]] = []
        for idx, r in enumerate(rows):
            f = await self.get_by_id(UUID(r["id"]))
            if f:
                results.append((f, 1.0 / (idx + 1)))
        return results

    async def update(self, fact: Fact) -> None:
        await self.add(fact)

    async def delete(self, fact_id: UUID) -> None:
        fid = str(fact_id)
        row = await self._db.fetchone(
            "SELECT vec_rowid FROM fact_rowid_map WHERE fact_id = ?", (fid,)
        )
        if row:
            rowid = int(row["vec_rowid"])
            await self._db.execute(
                "DELETE FROM facts_vec WHERE rowid = ?", (rowid,)
            )
            await self._db.execute(
                "DELETE FROM facts_fts_word WHERE rowid = ?", (rowid,)
            )
            await self._db.execute(
                "DELETE FROM facts_fts_tri WHERE rowid = ?", (rowid,)
            )
            await self._db.execute(
                "DELETE FROM fact_rowid_map WHERE fact_id = ?", (fid,)
            )
        await self._db.execute("DELETE FROM tags WHERE fact_id = ?", (fid,))
        await self._db.execute("DELETE FROM facts WHERE id = ?", (fid,))

    # ---------- 额外公共方法 ----------

    async def find_by_entity_ids(
        self, entity_ids: list[str], limit: int = 50
    ) -> list[tuple[Fact, float]]:
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        rows = await self._db.fetchall(
            f"""SELECT t.fact_id AS id, SUM(t.weight) AS w
                FROM tags t
                WHERE t.entity_id IN ({ph})
                GROUP BY t.fact_id
                ORDER BY w DESC
                LIMIT ?""",
            tuple(entity_ids) + (limit,),
        )
        result: list[tuple[Fact, float]] = []
        for r in rows:
            f = await self.get_by_id(UUID(r["id"]))
            if f:
                result.append((f, float(r["w"])))
        return result

    async def find_recent(self, limit: int = 50) -> list[Fact]:
        rows = await self._db.fetchall(
            """SELECT * FROM facts
               ORDER BY COALESCE(last_accessed_at, created_at) DESC
               LIMIT ?""",
            (limit,),
        )
        return [await self._row_to_fact(r) for r in rows]

    async def find_by_time_range(
        self, start: str | None, end: str | None, limit: int = 100
    ) -> list[Fact]:
        clauses = []
        params: list = []
        if start:
            clauses.append("happened_at >= ?")
            params.append(start)
        if end:
            clauses.append("happened_at <= ?")
            params.append(end)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        rows = await self._db.fetchall(
            f"SELECT * FROM facts {where} ORDER BY happened_at LIMIT ?", tuple(params)
        )
        return [await self._row_to_fact(r) for r in rows]

    async def list_all(self, limit: int = 1000) -> list[Fact]:
        rows = await self._db.fetchall(
            "SELECT * FROM facts ORDER BY created_at LIMIT ?", (limit,)
        )
        return [await self._row_to_fact(r) for r in rows]

    async def _row_to_fact(self, row) -> Fact:
        meta_dict = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        meta = Metadata(
            Person=meta_dict.get("Person", "") or "",
            Object=meta_dict.get("Object", "") or "",
            Location=meta_dict.get("Location", "") or "",
            Event=meta_dict.get("Event", "") or "",
            Organization=meta_dict.get("Organization", "") or "",
            Preference=meta_dict.get("Preference", "") or "",
            HappendTime=_parse(meta_dict.get("HappendTime")),
            MentionedTime=_parse(meta_dict.get("MentionedTime")),
            History=meta_dict.get("History"),
        )
        # Tags
        tag_rows = await self._db.fetchall(
            "SELECT entity_id, weight FROM tags WHERE fact_id = ?", (row["id"],)
        )
        tags: list[Tag] = []
        for tr in tag_rows:
            ent_row = await self._db.fetchone(
                "SELECT * FROM entities WHERE id = ?", (tr["entity_id"],)
            )
            if ent_row:
                from internal.domain.model.entity import Entity

                ent = Entity(
                    id=UUID(ent_row["id"]),
                    name=ent_row["name"],
                    centrality=ent_row["centrality"] or 0.0,
                )
                tags.append(Tag(Entity=ent, Weight=float(tr["weight"])))

        # Embedding（按 rowid 拉回）
        embedding: list[float] | None = None
        map_row = await self._db.fetchone(
            "SELECT vec_rowid FROM fact_rowid_map WHERE fact_id = ?", (row["id"],)
        )
        if map_row:
            vec_row = await self._db.fetchone(
                "SELECT embedding FROM facts_vec WHERE rowid = ?", (int(map_row["vec_rowid"]),)
            )
            if vec_row and vec_row["embedding"] is not None:
                import array as _array

                arr = _array.array("f")
                arr.frombytes(vec_row["embedding"])
                embedding = list(arr)

        return Fact(
            id=UUID(row["id"]),
            membox_id=UUID(row["membox_id"]) if row["membox_id"] else None,
            content=row["content"] or "",
            embedding=embedding,
            score=float(row["score"] or 0.0),
            tier=Tier(row["tier"] or "L0"),
            created_at=_parse(row["created_at"]) or datetime.now(timezone.utc),
            last_accessed_at=_parse(row["last_accessed_at"]),
            access_count=int(row["access_count"] or 0),
            original_msg=row["original_msg"] or "",
            metadata=meta,
            tag=tags,
        )

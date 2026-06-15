"""Entity 仓储的 SQLite 实现：FTS5 trigram + vec0 双索引 + RRF。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID

from internal.domain.model.entity import Entity
from internal.domain.repositories.entity_repository import EntityRepository
from internal.infra.database.sqlite import Database, serialize_embedding
from internal.util.rrf import rrf_merge

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteEntityRepository(EntityRepository):
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _get_or_assign_vec_rowid(self, entity_id: str) -> int:
        row = await self._db.fetchone(
            "SELECT vec_rowid FROM entity_rowid_map WHERE entity_id = ?", (entity_id,)
        )
        if row is not None and row["vec_rowid"] is not None:
            return int(row["vec_rowid"])
        # 获取下一个可用 rowid（sqlite implicit rowid 自增）
        max_row = await self._db.fetchone(
            "SELECT COALESCE(MAX(vec_rowid), 0) + 1 AS n FROM entity_rowid_map"
        )
        next_id = int(max_row["n"]) if max_row else 1
        await self._db.execute(
            "INSERT OR REPLACE INTO entity_rowid_map(entity_id, vec_rowid) VALUES (?,?)",
            (entity_id, next_id),
        )
        return next_id

    async def add(self, entity: Entity) -> None:
        eid = str(entity.id)
        await self._db.execute(
            """INSERT OR REPLACE INTO entities(id, name, centrality, created_at, updated_at)
               VALUES (?,?,?,?,?)""",
            (eid, entity.name, entity.centrality, _now(), _now()),
        )
        vec_rowid = await self._get_or_assign_vec_rowid(eid)
        # FTS5 内容表
        await self._db.execute("DELETE FROM entities_fts WHERE rowid = ?", (vec_rowid,))
        await self._db.execute(
            "INSERT INTO entities_fts(rowid, name) VALUES (?, ?)", (vec_rowid, entity.name)
        )
        if entity.embedding is not None:
            await self._db.execute(
                "DELETE FROM entities_vec WHERE rowid = ?", (vec_rowid,)
            )
            await self._db.execute(
                "INSERT INTO entities_vec(rowid, embedding) VALUES (?, ?)",
                (vec_rowid, serialize_embedding(entity.embedding)),
            )

    async def get_by_id(self, entity_id: UUID) -> Entity | None:
        row = await self._db.fetchone(
            "SELECT * FROM entities WHERE id = ?", (str(entity_id),)
        )
        return _row_to_entity(row) if row else None

    async def get_by_name(self, name: str) -> Entity | None:
        row = await self._db.fetchone(
            "SELECT * FROM entities WHERE name = ?", (name,)
        )
        return _row_to_entity(row) if row else None

    async def find_similar(
        self, embedding: list[float], top_k: int = 10
    ) -> list[Entity]:
        rows = await self._db.fetchall(
            """SELECT m.entity_id AS id
               FROM entities_vec v
               JOIN entity_rowid_map m ON m.vec_rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (serialize_embedding(embedding), top_k),
        )
        ids = [r["id"] for r in rows]
        return await self._fetch_entities(ids)

    async def upsert_by_name(self, entity: Entity) -> Entity:
        existing = await self.get_by_name(entity.name)
        if existing is not None:
            if entity.embedding is not None and existing.embedding is None:
                existing.embedding = entity.embedding
                await self.add(existing)
            return existing
        await self.add(entity)
        return entity

    async def find_by_names(self, names: list[str]) -> list[Entity]:
        if not names:
            return []
        ph = ",".join("?" * len(names))
        rows = await self._db.fetchall(
            f"SELECT * FROM entities WHERE name IN ({ph})", tuple(names)
        )
        return [_row_to_entity(r) for r in rows]

    # ---------- 额外公共方法（写入/读取管线会用）----------

    async def hybrid_recall(
        self,
        keywords: list[str],
        embedding: list[float] | None,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[Entity]:
        """Trigram + Embedding 双路召回 → RRF 合并。"""
        ranked: list[list[str]] = []

        # Trigram (FTS5)
        if keywords:
            fts_query = " OR ".join(f'"{kw}"' for kw in keywords if kw.strip())
            if fts_query:
                rows = await self._db.fetchall(
                    """SELECT m.entity_id AS id
                       FROM entities_fts f
                       JOIN entity_rowid_map m ON m.vec_rowid = f.rowid
                       WHERE entities_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, top_k),
                )
                ranked.append([r["id"] for r in rows])

        # Vector
        if embedding is not None:
            rows = await self._db.fetchall(
                """SELECT m.entity_id AS id
                   FROM entities_vec v
                   JOIN entity_rowid_map m ON m.vec_rowid = v.rowid
                   WHERE v.embedding MATCH ? AND k = ?
                   ORDER BY v.distance""",
                (serialize_embedding(embedding), top_k),
            )
            ranked.append([r["id"] for r in rows])

        if not ranked:
            return []
        merged = rrf_merge(ranked, k=rrf_k)[:top_k]
        ids = [eid for eid, _ in merged]
        return await self._fetch_entities(ids)

    async def _fetch_entities(self, ids: Iterable[str]) -> list[Entity]:
        ids = list(ids)
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        rows = await self._db.fetchall(
            f"SELECT * FROM entities WHERE id IN ({ph})", tuple(ids)
        )
        order = {eid: i for i, eid in enumerate(ids)}
        ents = [_row_to_entity(r) for r in rows]
        ents.sort(key=lambda e: order.get(str(e.id), 1_000_000))
        return ents


def _row_to_entity(row) -> Entity:
    return Entity(
        id=UUID(row["id"]),
        name=row["name"],
        centrality=row["centrality"] or 0.0,
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _parse_dt(s: str | None) -> datetime:
    if s is None:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(s)

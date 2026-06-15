"""Edge 仓储 SQLite 实现。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from internal.domain.model.edge import Edge
from internal.domain.repositories.edge_repository import EdgeRepository
from internal.infra.database.sqlite import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse(s: str | None) -> datetime | None:
    if s is None or s == "":
        return None
    return datetime.fromisoformat(s)


class SQLiteEdgeRepository(EdgeRepository):
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, edge: Edge) -> None:
        info_str = edge.info if isinstance(edge.info, str) else "RELATED_TO"
        history_json = json.dumps(edge.history, ensure_ascii=False) if edge.history else None
        await self._db.execute(
            """INSERT OR REPLACE INTO edges
               (id, from_fact_id, to_fact_id, info, weight, confidence,
                t_valid, t_invalid, created_at, updated_at, history_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(edge.id),
                str(edge.from_fact_id),
                str(edge.to_fact_id),
                info_str,
                edge.weight,
                getattr(edge, "confidence", 1.0),
                getattr(edge, "t_valid", None),
                getattr(edge, "t_invalid", None),
                _iso(edge.created_at) or _now(),
                _iso(edge.updated_at) or _now(),
                history_json,
            ),
        )

    async def find_by_fact(self, fact_id: UUID) -> list[Edge]:
        fid = str(fact_id)
        rows = await self._db.fetchall(
            "SELECT * FROM edges WHERE from_fact_id = ? OR to_fact_id = ?", (fid, fid)
        )
        return [_row_to_edge(r) for r in rows]

    async def update_weight(self, edge_id: UUID, weight: float) -> None:
        await self._db.execute(
            "UPDATE edges SET weight = ?, updated_at = ? WHERE id = ?",
            (weight, _now(), str(edge_id)),
        )

    async def delete_by_fact(self, fact_id: UUID) -> None:
        fid = str(fact_id)
        await self._db.execute(
            "DELETE FROM edges WHERE from_fact_id = ? OR to_fact_id = ?", (fid, fid)
        )

    # ---------- 额外公共方法 ----------

    async def find_top_neighbors(
        self, fact_id: UUID, top_n: int = 3
    ) -> list[Edge]:
        fid = str(fact_id)
        rows = await self._db.fetchall(
            """SELECT * FROM edges
               WHERE from_fact_id = ? OR to_fact_id = ?
               ORDER BY weight DESC LIMIT ?""",
            (fid, fid, top_n),
        )
        return [_row_to_edge(r) for r in rows]

    async def degree(self, fact_id: UUID) -> int:
        fid = str(fact_id)
        row = await self._db.fetchone(
            "SELECT COUNT(*) AS c FROM edges WHERE from_fact_id = ? OR to_fact_id = ?",
            (fid, fid),
        )
        return int(row["c"]) if row else 0

    async def max_degree(self) -> int:
        row = await self._db.fetchone(
            """SELECT MAX(c) AS m FROM (
                  SELECT COUNT(*) AS c FROM edges GROUP BY from_fact_id
                  UNION ALL
                  SELECT COUNT(*) AS c FROM edges GROUP BY to_fact_id
               )"""
        )
        return int(row["m"]) if row and row["m"] is not None else 1


def _row_to_edge(row) -> Edge:
    history = json.loads(row["history_json"]) if row["history_json"] else None
    e = Edge(
        id=UUID(row["id"]),
        from_fact_id=UUID(row["from_fact_id"]),
        to_fact_id=UUID(row["to_fact_id"]),
        info=row["info"] or "RELATED_TO",
        weight=float(row["weight"] or 0.5),
        created_at=_parse(row["created_at"]) or datetime.now(timezone.utc),
        updated_at=_parse(row["updated_at"]) or datetime.now(timezone.utc),
        history=history,
    )
    # 附加非模型字段
    setattr(e, "confidence", float(row["confidence"] or 1.0))
    setattr(e, "t_valid", row["t_valid"])
    setattr(e, "t_invalid", row["t_invalid"])
    return e

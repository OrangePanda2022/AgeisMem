"""MemBox 仓储 SQLite 实现。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from internal.domain.model.membox import MemBox
from internal.domain.repositories.membox_repository import MemBoxRepository
from internal.infra.database.sqlite import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse(s: str | None) -> datetime | None:
    if s is None or s == "":
        return None
    return datetime.fromisoformat(s)


class SQLiteMemBoxRepository(MemBoxRepository):
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(self, membox: MemBox) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO memboxes
               (id, title, summary, box_score, created_at, updated_at,
                last_accessed_at, access_count)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                str(membox.id),
                membox.title,
                membox.summary,
                membox.box_score,
                _iso(membox.created_at) or _now(),
                _iso(membox.updated_at) or _now(),
                _iso(membox.last_accessed_at),
                membox.access_count,
            ),
        )

    async def get_by_id(self, membox_id: UUID) -> MemBox | None:
        row = await self._db.fetchone(
            "SELECT * FROM memboxes WHERE id = ?", (str(membox_id),)
        )
        return _row_to_membox(row) if row else None

    async def find_by_time_range(
        self, start: str | None, end: str | None, limit: int = 100
    ) -> list[MemBox]:
        clauses = []
        params: list = []
        if start:
            clauses.append("created_at >= ?")
            params.append(start)
        if end:
            clauses.append("created_at <= ?")
            params.append(end)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)
        rows = await self._db.fetchall(
            f"SELECT * FROM memboxes {where} ORDER BY created_at LIMIT ?",
            tuple(params),
        )
        return [_row_to_membox(r) for r in rows]

    async def find_by_tier(self, tier: str, limit: int = 100) -> list[MemBox]:
        # MemBox 本身没有 tier 字段；按 box_score 区段近似
        rows = await self._db.fetchall(
            "SELECT * FROM memboxes ORDER BY box_score DESC LIMIT ?", (limit,)
        )
        return [_row_to_membox(r) for r in rows]

    async def list_all(self, offset: int = 0, limit: int = 50) -> list[MemBox]:
        rows = await self._db.fetchall(
            "SELECT * FROM memboxes ORDER BY created_at LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [_row_to_membox(r) for r in rows]

    async def update(self, membox: MemBox) -> None:
        await self.add(membox)

    async def delete(self, membox_id: UUID) -> None:
        await self._db.execute(
            "DELETE FROM memboxes WHERE id = ?", (str(membox_id),)
        )

    async def find_by_summary_keyword(self, keyword: str, limit: int = 10) -> list[MemBox]:
        rows = await self._db.fetchall(
            "SELECT * FROM memboxes WHERE summary LIKE ? OR title LIKE ? LIMIT ?",
            (f"%{keyword}%", f"%{keyword}%", limit),
        )
        return [_row_to_membox(r) for r in rows]


def _row_to_membox(row) -> MemBox:
    return MemBox(
        id=UUID(row["id"]),
        title=row["title"] or "",
        summary=row["summary"] or "",
        box_score=float(row["box_score"] or 0.0),
        created_at=_parse(row["created_at"]) or datetime.now(timezone.utc),
        updated_at=_parse(row["updated_at"]) or datetime.now(timezone.utc),
        last_accessed_at=_parse(row["last_accessed_at"]),
        access_count=int(row["access_count"] or 0),
    )

"""SQLite 连接管理 + schema 初始化（含 sqlite-vec 与 FTS5 trigram）。"""

from __future__ import annotations

import array
import asyncio
import logging
from pathlib import Path
from typing import Iterable

import aiosqlite
import sqlite_vec

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "memory.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  centrality REAL DEFAULT 0,
  created_at TEXT,
  updated_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
  name,
  tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING vec0(
  embedding float[2048]
);

CREATE TABLE IF NOT EXISTS memboxes (
  id TEXT PRIMARY KEY,
  title TEXT,
  summary TEXT,
  box_score REAL DEFAULT 0,
  created_at TEXT,
  updated_at TEXT,
  last_accessed_at TEXT,
  access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY,
  membox_id TEXT REFERENCES memboxes(id),
  content TEXT,
  original_msg TEXT,
  score REAL DEFAULT 0,
  tier TEXT DEFAULT 'L0',
  access_count INTEGER DEFAULT 0,
  created_at TEXT,
  last_accessed_at TEXT,
  metadata_json TEXT,
  happened_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_happened ON facts(happened_at);
CREATE INDEX IF NOT EXISTS idx_facts_tier ON facts(tier);
CREATE INDEX IF NOT EXISTS idx_facts_membox ON facts(membox_id);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
  embedding float[2048]
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts_word USING fts5(
  content, original_msg,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts_tri USING fts5(
  content, original_msg,
  tokenize='trigram'
);

CREATE TABLE IF NOT EXISTS tags (
  fact_id TEXT,
  entity_id TEXT,
  weight REAL,
  PRIMARY KEY(fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_tags_entity ON tags(entity_id);
CREATE INDEX IF NOT EXISTS idx_tags_fact ON tags(fact_id);

CREATE TABLE IF NOT EXISTS edges (
  id TEXT PRIMARY KEY,
  from_fact_id TEXT,
  to_fact_id TEXT,
  info TEXT,
  weight REAL DEFAULT 0.5,
  confidence REAL DEFAULT 1.0,
  t_valid TEXT,
  t_invalid TEXT,
  created_at TEXT,
  updated_at TEXT,
  history_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_fact_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_fact_id);

CREATE TABLE IF NOT EXISTS entity_rowid_map (
  entity_id TEXT PRIMARY KEY,
  vec_rowid INTEGER UNIQUE
);

CREATE TABLE IF NOT EXISTS fact_rowid_map (
  fact_id TEXT PRIMARY KEY,
  vec_rowid INTEGER UNIQUE
);
"""


def serialize_embedding(vec: Iterable[float]) -> bytes:
    """vec0 接受标准的 float32 小端字节流。"""
    return array.array("f", list(vec)).tobytes()


class Database:
    """异步 SQLite 连接持有者，模块单例。"""

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> str:
        return self._path

    async def connect(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is not None:
                return self._conn
            # aiosqlite 0.22 的 worker thread 默认非 daemon，调用方一旦漏 close()
            # 主进程就会卡在 threading._shutdown() 等线程退出（实测会无限挂）。
            # 必须在 .start() 之前（即 await connect() 之前）设 daemon=True。
            # aiosqlite.connect() 返回 Connection 代理，Thread 已构造但未 start，
            # __await__ 才会调 .start()。私有属性 _thread 在 0.20-0.22 都稳定。
            conn_proxy = aiosqlite.connect(self._path)
            thread = getattr(conn_proxy, "_thread", None)
            if thread is None:
                logger.warning(
                    "aiosqlite Connection has no _thread attribute (version=%s); "
                    "worker thread cannot be set daemon, process may hang on exit "
                    "if close() is missed",
                    getattr(aiosqlite, "__version__", "?"),
                )
            else:
                thread.daemon = True
            conn = await conn_proxy
            await conn.enable_load_extension(True)
            # sqlite_vec.load 是同步 C API，必须在 worker thread 里跑（不能直接
            # 在 event loop 调用，会污染 sqlite3 connection 的线程归属）。
            # aiosqlite 提供的 _execute 私有方法是官方 sqlite_vec docs 推荐写法。
            await conn._execute(sqlite_vec.load, conn._conn)  # type: ignore[attr-defined]
            await conn.enable_load_extension(False)
            conn.row_factory = aiosqlite.Row
            # synchronous=OFF：评测/单题场景下可接受（崩溃时丢未 fsync 的数据，
            # 但每题独立 db 即丢即弃）；生产环境应改 NORMAL。
            await conn.execute("PRAGMA synchronous=OFF")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            self._conn = conn
            logger.info("SQLite connected at %s", self._path)
            return conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def execute(self, sql: str, params: tuple = ()) -> None:
        conn = await self.connect()
        await conn.execute(sql, params)
        await conn.commit()

    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        conn = await self.connect()
        await conn.executemany(sql, params_list)
        await conn.commit()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        conn = await self.connect()
        async with conn.execute(sql, params) as cur:
            return list(await cur.fetchall())

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        conn = await self.connect()
        async with conn.execute(sql, params) as cur:
            return await cur.fetchone()

    async def execute_returning_rowid(self, sql: str, params: tuple = ()) -> int:
        conn = await self.connect()
        cur = await conn.execute(sql, params)
        rowid = cur.lastrowid
        await cur.close()
        await conn.commit()
        return int(rowid or 0)


_db_map: dict[str, Database] = {}


def get_db(path: str | None = None) -> Database:
    key = path or DEFAULT_DB_PATH
    if key not in _db_map:
        _db_map[key] = Database(key)
    return _db_map[key]


async def reset_db(path: str | None = None) -> Database:
    """删除并重新初始化（用于评测每题独立 db）。"""
    target = path or DEFAULT_DB_PATH
    existing = _db_map.pop(target, None)
    if existing is not None:
        await existing.close()
    p = Path(target)
    if p.exists():
        p.unlink()
    db = Database(target)
    await db.connect()
    _db_map[target] = db
    return db


if __name__ == "__main__":
    async def _main():
        db = get_db("memory.db")
        await db.connect()
        rows = await db.fetchall(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name"
        )
        print("Objects:", [r["name"] for r in rows])
        await db.close()

    asyncio.run(_main())

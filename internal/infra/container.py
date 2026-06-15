"""依赖容器：把数据库 + 仓库 + 客户端集中装配。"""

from __future__ import annotations

from dataclasses import dataclass

from internal.infra.database.sqlite import Database, get_db
from internal.infra.models.embedding.embedding import embedding_client
from internal.infra.models.llm.llm import llm_client
from internal.infra.repositories.sqlite_edge_repository import SQLiteEdgeRepository
from internal.infra.repositories.sqlite_entity_repository import SQLiteEntityRepository
from internal.infra.repositories.sqlite_fact_repository import SQLiteFactRepository
from internal.infra.repositories.sqlite_membox_repository import SQLiteMemBoxRepository


@dataclass
class Container:
    db: Database
    entities: SQLiteEntityRepository
    facts: SQLiteFactRepository
    edges: SQLiteEdgeRepository
    memboxes: SQLiteMemBoxRepository
    llm = llm_client
    embedder = embedding_client


def make_container(db_path: str | None = None) -> Container:
    db = get_db(db_path)
    return Container(
        db=db,
        entities=SQLiteEntityRepository(db),
        facts=SQLiteFactRepository(db),
        edges=SQLiteEdgeRepository(db),
        memboxes=SQLiteMemBoxRepository(db),
    )

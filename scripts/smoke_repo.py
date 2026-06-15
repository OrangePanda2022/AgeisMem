"""Smoke test: 重置 DB → 写 entity/fact/edge/membox → 召回 → 校验。"""

import asyncio
import random
from uuid import uuid4

from internal.domain.model.edge import Edge
from internal.domain.model.entity import Entity
from internal.domain.model.fact import Fact
from internal.domain.model.membox import MemBox
from internal.domain.model.tag import Tag
from internal.domain.model.tier import Tier
from internal.infra.database.sqlite import get_db, reset_db
from internal.infra.repositories.sqlite_edge_repository import SQLiteEdgeRepository
from internal.infra.repositories.sqlite_entity_repository import SQLiteEntityRepository
from internal.infra.repositories.sqlite_fact_repository import SQLiteFactRepository
from internal.infra.repositories.sqlite_membox_repository import SQLiteMemBoxRepository


def fake_embedding(seed: int) -> list[float]:
    random.seed(seed)
    return [random.random() for _ in range(2048)]


async def main():
    await reset_db("smoke.db")
    db = get_db("smoke.db")
    ents = SQLiteEntityRepository(db)
    facts = SQLiteFactRepository(db)
    edges = SQLiteEdgeRepository(db)
    boxes = SQLiteMemBoxRepository(db)

    # Entity
    e1 = Entity(name="花生", embedding=fake_embedding(1))
    e2 = Entity(name="过敏", embedding=fake_embedding(2))
    await ents.add(e1)
    await ents.add(e2)
    got = await ents.get_by_name("花生")
    assert got is not None and got.name == "花生"
    print("[OK] Entity add / get_by_name")

    sim = await ents.find_similar(fake_embedding(1), top_k=5)
    print(f"  similar returned: {[(s.name, str(s.id)[:8]) for s in sim]}; want e1.id={str(e1.id)[:8]}")
    assert any(str(s.id) == str(e1.id) for s in sim)
    print(f"[OK] Entity vec find_similar -> {[s.name for s in sim]}")

    hybrid = await ents.hybrid_recall(["花生"], fake_embedding(1), top_k=5)
    assert hybrid, "hybrid recall returned empty"
    print(f"[OK] Entity hybrid_recall (trigram+vec+RRF) -> {[h.name for h in hybrid]}")

    # MemBox
    box = MemBox(title="健康记录", summary="用户健康相关")
    await boxes.add(box)
    got_box = await boxes.get_by_id(box.id)
    assert got_box and got_box.title == "健康记录"
    print("[OK] MemBox add / get_by_id")

    # Fact
    f1 = Fact(
        membox_id=box.id,
        content="用户对花生过敏",
        embedding=fake_embedding(3),
        tier=Tier.L0,
        tag=[Tag(Entity=e1, Weight=0.9), Tag(Entity=e2, Weight=0.8)],
    )
    f2 = Fact(
        membox_id=box.id,
        content="用户喜欢绿茶",
        embedding=fake_embedding(4),
        tier=Tier.L0,
    )
    await facts.add(f1)
    await facts.add(f2)
    got_f = await facts.get_by_id(f1.id)
    assert got_f is not None and got_f.content == "用户对花生过敏"
    assert len(got_f.tag) == 2
    print(f"[OK] Fact add / get_by_id (tags={len(got_f.tag)})")

    vec_hits = await facts.vector_search(fake_embedding(3), top_k=3)
    assert any(str(h.id) == str(f1.id) for h in vec_hits)
    print(f"[OK] Fact vector_search -> {[h.content for h in vec_hits]}")

    tag_hits = await facts.find_by_entity_ids([str(e1.id)])
    assert tag_hits and str(tag_hits[0][0].id) == str(f1.id)
    print(f"[OK] Fact find_by_entity_ids -> {[(f.content, w) for f, w in tag_hits]}")

    recent = await facts.find_recent(limit=5)
    assert len(recent) == 2
    print(f"[OK] Fact find_recent -> {len(recent)}")

    # Edge
    edge = Edge(from_fact_id=f1.id, to_fact_id=f2.id, info="RELATED_TO", weight=0.7)
    await edges.add(edge)
    nbrs = await edges.find_by_fact(f1.id)
    assert nbrs and nbrs[0].weight == 0.7
    print(f"[OK] Edge add / find_by_fact -> {nbrs[0].info} w={nbrs[0].weight}")

    deg = await edges.degree(f1.id)
    assert deg == 1
    print(f"[OK] Edge degree -> {deg}")

    await db.close()
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

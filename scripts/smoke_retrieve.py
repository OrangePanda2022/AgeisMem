"""Retrieval pipeline smoke test。

流程：
  1. 重置 DB
  2. 写入 3 条对话 → 检查 Fact
  3. 用查询检索 → 检查召回
  4. 完整 answer 管线 → 检查输出
"""

import asyncio
from datetime import datetime, timezone

from internal.infra.container import make_container
from internal.infra.database.sqlite import reset_db
from internal.service.retrieve.recall import RecallService
from internal.service.retrieve.manager import MASComputeService
from internal.service.retrieve.cba import CBAService
from main import MemoryRetrievalPipeline


async def main():
    print("==> Smoke test: retrieval pipeline")
    await reset_db("smoke_retrieve.db")
    pipeline = MemoryRetrievalPipeline(db_path="smoke_retrieve.db")

    # 1) ingest
    print("\n==> Ingesting messages...")
    await pipeline.ingest(
        "我对花生过敏，请避开花生制品。",
        event_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    await pipeline.ingest(
        "我喜欢喝绿茶，咖啡就不要了。",
        event_time=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    await pipeline.ingest(
        "我养了一只叫小白的猫。",
        event_time=datetime(2026, 6, 3, tzinfo=timezone.utc),
    )

    all_facts = await pipeline.container.facts.list_all()
    print(f"  total facts in DB: {len(all_facts)}")
    for f in all_facts:
        print(f"    - {f.content} | tags={[t.Entity.name for t in f.tag]}")

    # 2) recall test
    print("\n==> Recall test: '过敏'")
    container = make_container("smoke_retrieve.db")
    recall_svc = RecallService(container)
    query_emb = await container.embedder.embed_text("过敏")
    recall_result = await recall_svc.recall("过敏", query_emb)
    print(f"  recalled {len(recall_result.facts)} facts")
    for f in recall_result.facts:
        print(f"    - {f.content}")
    assert any("过敏" in f.content for f in recall_result.facts), "Should recall allergy fact"
    print("  [OK] recall finds allergy fact")

    # 3) MAS scoring test
    print("\n==> MAS scoring test")
    mas_svc = MASComputeService(container)
    scored = await mas_svc.compute_mas_scores(query_emb, recall_result.facts)
    print(f"  scored {len(scored)} facts")
    for f, s in scored:
        print(f"    - {f.content[:30]} | MAS={s:.4f}")
    assert scored[0][1] > 0, "Top fact should have positive MAS"
    print("  [OK] MAS scoring works")

    # 4) CBA test
    print("\n==> CBA test")
    cba_svc = CBAService()
    budgeted = await cba_svc.allocate(scored, total_budget=2000)
    context = await cba_svc.build_retrieval_context(budgeted, pipeline.buffer)
    print(f"  context length: {len(context)} chars")
    assert "过敏" in context, "Context should contain allergy info"
    print("  [OK] CBA builds context")

    # 5) Full answer pipeline
    print("\n==> Answer pipeline: '用户对什么过敏？'")
    answer = await pipeline.answer("用户对什么过敏？")
    print(f"  answer: {answer}")
    assert len(answer) > 0, "Should produce non-empty answer"
    print("  [OK] Full pipeline produces answer")

    await container.db.close()
    print("\nAll retrieval smoke checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
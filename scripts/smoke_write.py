"""Smoke test 写入管线：注入 2 条消息，应在 DB 中看到 Fact/Tag/Edge。"""

import asyncio
from datetime import datetime, timezone

from internal.infra.container import make_container
from internal.infra.database.sqlite import reset_db
from internal.service.input.write_service import WriteService


async def main():
    await reset_db("write_smoke.db")
    c = make_container("write_smoke.db")
    ws = WriteService(c)

    print("==> ingest 1")
    out1 = await ws.ingest(
        "我对花生过敏，请避开花生制品。",
        event_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    print(f"  wrote {len(out1)} facts: {[f.content for f in out1]}")

    print("==> ingest 2 (related)")
    out2 = await ws.ingest(
        "我喜欢喝绿茶，咖啡就不要了。",
        event_time=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    print(f"  wrote {len(out2)} facts: {[f.content for f in out2]}")

    print("==> ingest 3 (conflict候选: 改口想吃花生)")
    out3 = await ws.ingest(
        "其实我不过敏花生了，最近脱敏治疗了一下。",
        event_time=datetime(2026, 6, 9, tzinfo=timezone.utc),
    )
    print(f"  wrote {len(out3)} facts: {[f.content for f in out3]}")

    all_facts = await c.facts.list_all()
    print(f"\n[DB] total facts: {len(all_facts)}")
    for f in all_facts:
        print(f"  - {f.content[:50]} | tier={f.tier.value} | tags={[t.Entity.name for t in f.tag]}")

    # 边
    for f in all_facts:
        nbrs = await c.edges.find_by_fact(f.id)
        if nbrs:
            print(f"  edges from/to {f.content[:30]}:")
            for e in nbrs:
                print(f"    {e.info} w={e.weight}")

    await c.db.close()


if __name__ == "__main__":
    asyncio.run(main())

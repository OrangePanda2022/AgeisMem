"""LLM Prompts smoke test - 验证 4 个 prompt 都能返回有效 JSON。"""

import asyncio
import json

from internal.infra.models.llm.llm import llm_client


async def main():
    # 1) FACT_EXTRACTION
    print("==> extract_facts")
    out = await llm_client.extract_facts(
        "我对花生过敏，请避开花生制品。", event_time="2026-06-09"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2)[:500])
    assert "facts" in out and isinstance(out["facts"], list)

    # 2) ENTITY_EXTRACTION
    print("\n==> extract_entities")
    ents = await llm_client.extract_entities("我搬到上海了，去年开始学法语。")
    print(ents)
    assert isinstance(ents, list)

    # 3) EVOLUTION_DECISION
    print("\n==> evolution_decision")
    plan = await llm_client.evolution_decision(
        fact_content="用户买了特斯拉 Model Y",
        membox_context={"title": "车辆", "summary": "用户的车辆变化"},
        existing_facts=[
            {"id": "11111111-1111-1111-1111-111111111111", "content": "用户开丰田", "edges": []}
        ],
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2)[:500])
    assert "decision" in plan

    # 4) TOPIC_LOOM
    print("\n==> topic_loom")
    km = await llm_client.topic_loom(
        fact_content="用户现在开什么车？",
        memboxes=[
            {"content": "用户卖掉了丰田", "edges": [{"info": "CONTRADICTS"}]},
            {"content": "用户买了特斯拉 Model Y", "edges": []},
        ],
    )
    print(json.dumps(km, ensure_ascii=False, indent=2)[:500])
    assert "key_memory" in km

    print("\nAll prompts smoke checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

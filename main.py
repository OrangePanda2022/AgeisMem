"""
AAAI Long-Term Memory System — Main Entry Point.

Orchestrates the full memory pipeline:
  ingest → forget → recall → MAS score → CBA → answer

CLI usage:
  python main.py ingest "message"           — write a message to memory
  python main.py answer "query"             — answer using memory
  python main.py eval --data <path>         — run LongMemEval evaluation
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from internal.config.settings import settings
from internal.domain.model.buffer import MessageBuffer
from internal.infra.container import Container, make_container
from internal.infra.database.sqlite import reset_db
from internal.service.forget.forget import ForgettingService
from internal.service.input.write_service import WriteService
from internal.service.retrieve.cba import CBAService
from internal.service.retrieve.manager import MASComputeService
from internal.service.retrieve.recall import RecallService
from internal.util.debug_collector import DebugCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class MemoryRetrievalPipeline:
    """完整记忆管线编排器。"""

    def __init__(self, container: Container | None = None, db_path: str = "memory.db") -> None:
        self.db_path = db_path
        self.container = container or make_container(db_path)
        self.recall = RecallService(self.container)
        self.mas = MASComputeService(self.container)
        self.cba = CBAService()
        self.buffer = MessageBuffer(max_size=20)
        self._forgetting = ForgettingService(self.container)

    async def ingest(
        self,
        message: str,
        event_time: datetime | None = None,
        role: str = "user",
    ) -> None:
        """写入一条消息到记忆系统，同时更新对话缓冲区。

        role="user" 视为用户陈述；role="assistant" 视为助手提供的信息
        （如知识/推荐/事实陈述），同样进入 fact store，仅在原文前加
        [Speaker: assistant] 标记，便于 LLM 抽取出第三人称事实。
        """
        write_svc = WriteService(self.container)
        if role == "assistant":
            tagged = f"[Speaker: assistant]\n{message}"
        else:
            tagged = message
        facts = await write_svc.ingest(tagged, event_time)
        logger.info("Ingested %d facts from %s message", len(facts), role)
        self.buffer.add(role if role in ("user", "assistant") else "user", message)

    async def answer(
        self,
        query: str,
        reference_time: datetime | None = None,
        *,
        debug_path: str | None = None,
        qid: str = "ad-hoc",
    ) -> str:
        """基于记忆检索回答用户问题。

        完整流程：
          query → forget → embed → recall(含投机召回) → MAS →
          [迭代召回: 充分性检查 → 查询扩展 → 定向召回] →
          CBA → [辩论模式/单轮] → LLM answer
        """
        debug = DebugCollector(qid=qid, query=query) if debug_path else None

        # 0) 召回前先执行遗忘衰减
        try:
            await self._forgetting.run_forgetting_cycle(
                reference_time=reference_time, debug=debug,
            )
        except Exception as e:
            logger.warning("Forgetting cycle failed: %s", e)

        # 1) 嵌入查询
        query_emb = await self.container.embedder.embed_text(query)
        if debug is not None:
            debug.record("query_embed", {
                "dim": len(query_emb),
                "head5": [float(x) for x in query_emb[:5]],
            })

        # 2) 召回（含方案B投机召回）
        recall_result = await self.recall.recall(query, query_emb, debug=debug)
        if not recall_result.facts:
            logger.info("No facts recalled for query: %s", query[:40])
            if debug is not None:
                debug.record("llm_answer", {"skipped": "no facts recalled"})
                debug.dump(debug_path)
            return "信息不足，无法回答。"

        logger.info("Recalled %d facts (%d expanded via graph walk)",
                     len(recall_result.facts), len(recall_result.expanded_facts))

        # 3) MAS 评分
        scored = await self.mas.compute_mas_scores(
            query_emb, recall_result.facts,
            reference_time=reference_time, debug=debug,
        )
        if not scored:
            if debug is not None:
                debug.record("llm_answer", {"skipped": "no scored facts"})
                debug.dump(debug_path)
            return "信息不足，无法回答。"

        # ---- 反向实体过滤（P2-3）：MAS 后检查 top-N 是否含 GT 预期实体 ----
        # 若 LLM 推断的 missing_entities 在 top 召回中缺失，触发第二轮 keyword 召回扩展。
        # 守卫：只当缺失实体 ≥ 2 且 top-10 里没有任何 fact 的 MAS ≥ 0.5 时才触发，
        # 避免 preference 类问题上 extract_expected_entities 幻觉导致假阳性。
        try:
            top_for_check = min(10, len(scored))
            check_summary_parts = []
            for fact, mas_score in scored[:top_for_check]:
                check_summary_parts.append(
                    f"- [{mas_score:.3f}] {fact.content[:150]}"
                )
            check_summary = "\n".join(check_summary_parts)
            expected = await self.container.llm.extract_expected_entities(
                query, check_summary, debug=debug,
            )
            missing_entities = expected.get("missing_entities", []) or []
            max_top_mas = max((s for _, s in scored[:top_for_check]), default=0.0)
            should_trigger = (
                len(missing_entities) >= 2
                and max_top_mas < 0.5
            )
            if missing_entities and not should_trigger:
                logger.info(
                    "Missing entities %s but guard skipped reverse expansion "
                    "(count=%d, max_top_mas=%.3f)",
                    missing_entities, len(missing_entities), max_top_mas,
                )
            if should_trigger:
                logger.info(
                    "Missing entities in top-%d: %s (max_top_mas=%.3f) — triggering reverse expansion",
                    top_for_check, missing_entities, max_top_mas,
                )
                # 用缺失实体作为新 keywords 跑迭代召回
                existing_ids = {str(f.id) for f, _ in scored}
                expand_result = await self.recall.recall_by_keywords(
                    missing_entities, query_emb,
                    existing_fact_ids=existing_ids, debug=debug,
                )
                if expand_result.facts:
                    logger.info(
                        "Reverse expansion: found %d new facts (%d expanded)",
                        len(expand_result.facts), len(expand_result.expanded_facts),
                    )
                    recall_result.merge(expand_result)
                    scored = await self.mas.compute_mas_scores(
                        query_emb, recall_result.facts,
                        reference_time=reference_time, debug=debug,
                    )
                    # 记录反向扩展触发
                    if debug is not None:
                        debug.record("reverse_entity_expansion", {
                            "missing_entities": missing_entities,
                            "new_facts_count": len(expand_result.facts),
                            "expanded_count": len(expand_result.expanded_facts),
                            "triggered": True,
                            "max_top_mas": max_top_mas,
                        })
                else:
                    if debug is not None:
                        debug.record("reverse_entity_expansion", {
                            "missing_entities": missing_entities,
                            "new_facts_count": 0,
                            "triggered": True,
                            "max_top_mas": max_top_mas,
                            "note": "no new facts found from missing entities",
                        })
            else:
                if debug is not None:
                    debug.record("reverse_entity_expansion", {
                        "missing_entities": missing_entities,
                        "triggered": False,
                        "max_top_mas": max_top_mas,
                    })
        except Exception as e:
            logger.warning("Reverse entity expansion failed: %s", e)
            if debug is not None:
                debug.record("reverse_entity_expansion", {
                    "error": str(e), "triggered": False,
                })
        # ---- 反向实体过滤结束 ----

        # ---- 方案A：迭代检索（充分性检查 + 查询扩展） ----
        if settings.iterative_retrieval_enabled:
            max_rounds = settings.iterative_retrieval_max_rounds
            for round_idx in range(max_rounds):
                # 构建紧凑摘要
                top_n = min(10, len(scored))
                summary_parts = []
                for fact, mas_score in scored[:top_n]:
                    tags_str = ", ".join(
                        t.Entity.name for t in fact.tag[:3]
                    ) if fact.tag else ""
                    pref = fact.metadata.Preference if fact.metadata else ""
                    meta_line = f" [Preference={pref}]" if pref else ""
                    tag_line = f" (tags: {tags_str})" if tags_str else ""
                    summary_parts.append(
                        f"- [{mas_score:.3f}] {fact.content[:120]}{meta_line}{tag_line}"
                    )
                context_summary = "\n".join(summary_parts)

                # 检查充分性
                try:
                    suff_result = await self.container.llm.check_sufficiency(
                        query, context_summary, debug=debug,
                    )
                except Exception as e:
                    logger.warning("Sufficiency check failed (round %d): %s", round_idx + 1, e)
                    break

                is_sufficient = suff_result.get("sufficient", True)
                alt_keywords = list(suff_result.get("alternative_keywords", []))

                # ---- P3-1：从 graph walk 邻居 fact 的 tag 实体扩展 alt_keywords ----
                try:
                    seed_facts = [f for f, _ in scored[:5]]
                    graph_kw = await self.recall.expand_keywords_via_graph(
                        seed_facts, debug=debug,
                    )
                    if graph_kw:
                        alt_keywords.extend(graph_kw)
                        logger.info(
                            "Round %d: graph-walk expanded keywords: %s",
                            round_idx + 1, graph_kw[:5],
                        )
                except Exception as e:
                    logger.warning("expand_keywords_via_graph failed (round %d): %s", round_idx + 1, e)

                # 强制 alt_keywords 与原 query 表面词不同；否则去掉重复项
                query_lower = query.lower()
                query_words = set(query_lower.split())
                seen_kw = set()
                deduped_kw = []
                for kw in alt_keywords:
                    kw_s = str(kw).strip()
                    if not kw_s:
                        continue
                    kw_l = kw_s.lower()
                    # 跳过与原 query 完全相同或仅 query 中已有词的 keyword
                    if kw_l in query_lower:
                        continue
                    if kw_l in query_words:
                        continue
                    if kw_l in seen_kw:
                        continue
                    seen_kw.add(kw_l)
                    deduped_kw.append(kw_s)
                alt_keywords = deduped_kw

                logger.info(
                    "Sufficiency check round %d: sufficient=%s, confidence=%.2f, keywords=%s (deduped from %d)",
                    round_idx + 1, is_sufficient,
                    suff_result.get("confidence", 0), alt_keywords,
                    len(suff_result.get("alternative_keywords", [])),
                )

                if is_sufficient or not alt_keywords:
                    break

                # 定向召回
                existing_ids = {str(f.id) for f, _ in scored}
                new_result = await self.recall.recall_by_keywords(
                    alt_keywords, query_emb,
                    existing_fact_ids=existing_ids, debug=debug,
                )

                if not new_result.facts:
                    logger.info("Iterative recall round %d: no new facts found", round_idx + 1)
                    break

                logger.info(
                    "Iterative recall round %d: found %d new facts (%d expanded)",
                    round_idx + 1, len(new_result.facts), len(new_result.expanded_facts),
                )

                # 合并 + 重新 MAS 排序
                recall_result.merge(new_result)
                scored = await self.mas.compute_mas_scores(
                    query_emb, recall_result.facts,
                    reference_time=reference_time, debug=debug,
                )
                if not scored:
                    break
        # ---- 迭代检索结束 ----

        logger.info("Top-3 MAS: %s",
                     [(f.content[:20], round(s, 3)) for f, s in scored[:3]])

        # 4) 预算分配
        budgeted = await self.cba.allocate(scored, debug=debug)

        # 偏好类问题：限制非偏好 facts 数量，避免偏好信息被淹没
        is_preference = any(
            kw in query.lower() for kw in
            ["recommend", "suggest", "tips", "advice", "prefer",
             "any ideas", "what should", "should i", "any tips",
             "what kind", "any good", "looking for",
             "应该", "推荐", "建议", "偏好", "有什么好"]
        )
        if is_preference:
            pref_budgeted = [(f, b) for f, b in budgeted
                             if f.metadata and f.metadata.Preference]
            other_budgeted = [(f, b) for f, b in budgeted
                              if not (f.metadata and f.metadata.Preference)]
            # Keep all preference facts, cap others at 5
            if len(pref_budgeted) > 0 and len(other_budgeted) > 5:
                other_budgeted = other_budgeted[:5]
                budgeted = pref_budgeted + other_budgeted
                logger.info("Preference query: capped non-preference facts to 5, "
                            "pref=%d other=%d total=%d",
                            len(pref_budgeted), len(other_budgeted), len(budgeted))

        # ---- P1-A: 上下文硬上限 — 避免长上下文淹没 top-MAS 信号 ----
        # 现象：b0479f84/fca70973 等题 ctx≥50 facts，LLM 注意力被低相关 fact 抢走，
        # top-3 MAS 命中的具体实体（如 "Brandon Flowers"）反而没写进 hypothesis。
        # budgeted 已按 MAS 降序排列，直接取前 12 即可保留最相关 facts。
        MAX_CONTEXT_FACTS = 12
        if len(budgeted) > MAX_CONTEXT_FACTS:
            logger.info("P1-A: hard capping context %d -> %d facts",
                        len(budgeted), MAX_CONTEXT_FACTS)
            budgeted = budgeted[:MAX_CONTEXT_FACTS]

        # 5) 构建上下文
        context = await self.cba.build_retrieval_context(
            budgeted, self.buffer, debug=debug,
        )

        # 6) 生成答案（方案C：偏好类问题启用辩论模式）
        is_preference = any(
            kw in query.lower() for kw in
            ["recommend", "suggest", "tips", "advice", "prefer",
             "any ideas", "what should", "should i", "any tips",
             "what kind", "any good", "looking for",
             "应该", "推荐", "建议", "偏好", "有什么好"]
        )
        use_debate = settings.debate_mode_enabled and is_preference

        try:
            if use_debate:
                logger.info("Using debate mode for preference query")
                answer_result = await self.container.llm.debate_answer(
                    query, context,
                    reference_time=reference_time.isoformat() if reference_time else None,
                    debug=debug,
                )
            else:
                answer_result = await self.container.llm.generate_answer(
                    query, context,
                    reference_time=reference_time.isoformat() if reference_time else None,
                    debug=debug,
                )
            answer_text = answer_result.get("answer", "信息不足，无法回答。")
        except Exception as e:
            logger.warning("LLM answer generation failed: %s", e)
            top_fact = scored[0][0].content if scored else "无相关记忆"
            answer_text = f"基于记忆：{top_fact}"
            if debug is not None:
                debug.record("llm_answer", {"error": str(e), "fallback": answer_text})
        finally:
            if debug is not None:
                debug.dump(debug_path)
        return answer_text

    async def answer_without_ingest(self, query: str) -> str:
        """仅检索不写入（用于评测场景）。"""
        return await self.answer(query)


async def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = MemoryRetrievalPipeline(db_path=args.db or "memory.db")
    await pipeline.ingest(args.message)
    print(f"Ingested: {args.message}")


async def cmd_answer(args: argparse.Namespace) -> None:
    pipeline = MemoryRetrievalPipeline(db_path=args.db or "memory.db")
    answer = await pipeline.answer(args.query)
    print(json.dumps({"query": args.query, "answer": answer}, ensure_ascii=False))


async def cmd_eval(args: argparse.Namespace) -> None:
    """LongMemEval 评测入口。"""
    results: list[dict] = []

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data[: args.max] if args.max else data
    logger.info("Running eval on %d items from %s", len(items), args.data)

    for i, entry in enumerate(items):
        qid = entry["question_id"]
        logger.info("[%d/%d] %s", i + 1, len(items), qid)

        # 每个问题独立 DB
        db_path = f"/tmp/eval_{qid.replace('/', '_')}.db"
        await reset_db(db_path)
        pipeline = MemoryRetrievalPipeline(db_path=db_path)

        # 回放所有对话 Session
        sessions = entry.get("haystack_sessions", [])
        dates = entry.get("haystack_dates", [])
        sess_ids = entry.get("haystack_session_ids", [])

        for sid, sess, date_str in zip(sess_ids, sessions, dates):
            try:
                event_time = datetime.fromisoformat(date_str) if date_str else None
            except Exception:
                event_time = None
            # 将 session 中的每一条消息写入
            for turn in sess:
                if turn["role"] == "user":
                    await pipeline.ingest(turn["content"], event_time=event_time, role="user")
                elif turn["role"] == "assistant":
                    await pipeline.ingest(turn["content"], event_time=event_time, role="assistant")

        # 回答问题
        hypothesis = await pipeline.answer(entry["question"])

        results.append({
            "question_id": qid,
            "hypothesis": hypothesis,
        })

    # 写入输出
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Eval results saved to %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="AAAI Memory System")
    parser.add_argument("--db", help="SQLite database path", default=None)

    subparsers = parser.add_subparsers(dest="command")

    ingest_p = subparsers.add_parser("ingest", help="Write a message to memory")
    ingest_p.add_argument("message", type=str)

    answer_p = subparsers.add_parser("answer", help="Answer a query using memory")
    answer_p.add_argument("query", type=str)

    eval_p = subparsers.add_parser("eval", help="Run LongMemEval evaluation")
    eval_p.add_argument("--data", type=str, required=True, help="Path to LongMemEval JSON")
    eval_p.add_argument("--output", type=str, default="/tmp/eval_results.jsonl", help="Output path")
    eval_p.add_argument("--max", type=int, default=None, help="Max questions")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(cmd_ingest(args))
    elif args.command == "answer":
        asyncio.run(cmd_answer(args))
    elif args.command == "eval":
        asyncio.run(cmd_eval(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
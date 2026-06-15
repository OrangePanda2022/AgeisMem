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

        完整流程：query → forget → embed → recall(时间+关键字+图游走) → MAS score → CBA → LLM answer。

        Args:
            query: 用户问题。
            reference_time: 用于遗忘衰减的"现在"参考点；评测时传入数据集
                最新事件时间，避免墙钟差异让全量 fact 归零。
            debug_path: 若给出，会把每步召回/打分/上下文/LLM 输入输出 dump 到该路径。
            qid: 调试标识，写入 debug 文件元数据。
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

        # 2) 召回
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

        logger.info("Top-3 MAS: %s",
                     [(f.content[:20], round(s, 3)) for f, s in scored[:3]])

        # 4) 预算分配
        budgeted = await self.cba.allocate(scored, debug=debug)

        # 5) 构建上下文
        context = await self.cba.build_retrieval_context(
            budgeted, self.buffer, debug=debug,
        )

        # 6) LLM 合成答案
        try:
            answer_result = await self.container.llm.generate_answer(
                query, context, debug=debug,
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
"""LongMemEval 评测运行器。

从本地 LongMemEval 数据目录读取 → 通过 aaai 记忆系统处理 → 输出 hypothesis JSONL。

用法：
  PYTHONPATH=. uv run python scripts/evaluate_longmemeval.py \\
      --data longmemeval_oracle.json \\
      --output ~/tmp/results.jsonl \\
      --max 10

如果 --data 给的是相对路径或仅文件名，会去 --data-dir 下查找；
默认数据目录为仓库同级 ../LongMemEval/data（由 __file__ 推导，跨机器可移植）。

评测脚本特性：
  - checkpoint：output 文件中已有的 question_id 会被跳过（append 模式）
  - reference_time：使用题目最大 haystack date 作为遗忘衰减锚点
  - 资源清理：每题完成后关闭 DB 连接并删除 ~/tmp/eval_*.db
  - 容错：单题异常/超时不影响其他题，错误记入 errors.jsonl
  - LLM/Embedding 限流由 settings.llm_max_concurrency 等控制（客户端层信号量）
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("longmemeval")


# 路径基于 __file__ 推导，相对本仓库根目录，避免硬编码机器相关绝对路径。
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = str(_REPO_ROOT.parent / "LongMemEval" / "data")
DEFAULT_TMP_DIR = str(Path.home() / "tmp")
DEFAULT_PER_ITEM_TIMEOUT = 720  # 单题超时（秒）


def _resolve_data_path(data_arg: str, data_dir: str) -> Path:
    """支持传入绝对路径 / 相对路径 / 仅文件名三种形式。"""
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return p
    candidate = Path(data_dir) / data_arg
    if candidate.exists():
        return candidate
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(
        f"Data file not found: {data_arg!r} (looked at {p} and {candidate})"
    )


def _load_completed_qids(output_path: str) -> set[str]:
    """从已有 output jsonl 中读出已完成的 question_id 集合。"""
    done: set[str] = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("question_id")
                if isinstance(qid, str):
                    done.add(qid)
            except json.JSONDecodeError:
                continue
    return done


def _max_haystack_time(dates: list[str]) -> datetime | None:
    """取 haystack_dates 的最大值作为遗忘衰减参考时间。"""
    parsed: list[datetime] = []
    for d in dates or []:
        dt = _parse_lme_date(d)
        if dt is not None:
            parsed.append(dt)
    return max(parsed) if parsed else None


# LongMemEval 的 haystack_dates 格式形如 "2023/04/10 (Mon) 17:50"，
# 不是合法 ISO8601；fromisoformat 会全部失败，导致事件时间退化为墙钟。
_LME_DATE_RE = re.compile(
    r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s*(?:\([^)]*\))?\s*"
    r"(?:(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?\s*$"
)


def _parse_lme_date(s: str | None) -> datetime | None:
    if not s:
        return None
    m = _LME_DATE_RE.match(s)
    if m:
        y, mo, d, h, mi, se = m.groups()
        # 统一打 UTC 标签：haystack_dates 没带时区，但 Fact.created_at 默认
        # datetime.now(timezone.utc)，混 naive/aware 在 graph_walk 里会 TypeError。
        return datetime(
            int(y), int(mo), int(d),
            int(h or 0), int(mi or 0), int(se or 0),
            tzinfo=timezone.utc,
        )
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval Evaluation Runner")
    parser.add_argument("--data", type=str, required=True,
                        help="LongMemEval JSON 文件名或路径（如 longmemeval_oracle.json）")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"本地 LongMemEval 数据目录（默认 {DEFAULT_DATA_DIR}）")
    parser.add_argument("--output", type=str, default=str(Path(DEFAULT_TMP_DIR) / "longmemeval_results.jsonl"),
                        help="Output JSONL file for hypotheses（append 模式，已有 qid 自动跳过）")
    parser.add_argument("--errors", type=str, default=None,
                        help="错误记录文件，默认与 --output 同目录的 errors.jsonl")
    parser.add_argument("--tmp-dir", type=str, default=DEFAULT_TMP_DIR,
                        help=f"评测期间的临时 SQLite 文件目录（默认 {DEFAULT_TMP_DIR}）")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of questions to evaluate")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="题目级并发；下游 LLM/Embedding 由 settings 限流")
    parser.add_argument("--per-item-timeout", type=int, default=DEFAULT_PER_ITEM_TIMEOUT,
                        help=f"单题超时秒数（默认 {DEFAULT_PER_ITEM_TIMEOUT}）")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已存在的 output 文件，从头跑（会覆盖）")
    parser.add_argument("--call-log", type=str, default=None,
                        help="单题调用追踪 JSONL（默认与 --output 同目录的 call_log.jsonl，append 模式）")
    args = parser.parse_args()

    try:
        data_path = _resolve_data_path(args.data, args.data_dir)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    args.data = str(data_path)

    logger.info("Loading LongMemEval data from %s", args.data)

    # 延迟导入 main 模块（避免循环导入）
    from main import MemoryRetrievalPipeline
    from internal.infra.database.sqlite import reset_db, _db_map

    # 流式加载:s_cleaned/m_cleaned 可达 2.6GB,json.load 会 OOM;
    # 用 ijson 逐对象解析(峰值仅约单题大小 + 物化后的列表)。
    # 物化成 list 以保留下游 len()/enumerate/[:max] 语义不变。
    import ijson
    data: list[dict] = []
    with open(args.data, "rb") as f:
        for obj in ijson.items(f, "item"):
            data.append(obj)

    items = data[:args.max] if args.max else data

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = Path(args.errors) if args.errors else output_path.with_name("errors.jsonl")
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = Path(args.call_log) if args.call_log else output_path.with_name("call_log.jsonl")

    if args.no_resume and output_path.exists():
        output_path.unlink()
    if args.no_resume and call_log_path.exists():
        call_log_path.unlink()
    done_qids = _load_completed_qids(str(output_path))
    # call_log 与 results 共享 resume 跳过集：任一文件含 qid 即视为已完成，避免重复追踪
    done_call_qids = _load_completed_qids(str(call_log_path))
    skipped_qids = done_qids | done_call_qids
    if done_qids:
        logger.info("Resume mode: %d question_id already in %s, will skip",
                    len(done_qids), output_path)
    if done_call_qids:
        logger.info("Resume mode: %d question_id already in %s, will skip",
                    len(done_call_qids), call_log_path)

    pending = [(i, e) for i, e in enumerate(items) if e.get("question_id") not in skipped_qids]
    logger.info("Evaluating %d items (%d total, %d skipped)",
                len(pending), len(items), len(items) - len(pending))

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    error_lock = asyncio.Lock()
    call_log_lock = asyncio.Lock()
    completed = 0

    async def write_result(record: dict) -> None:
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def write_call_log(record: dict) -> None:
        async with call_log_lock:
            with call_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def write_error(record: dict) -> None:
        async with error_lock:
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def process_one(entry: dict, idx: int) -> None:
        nonlocal completed
        qid = entry["question_id"]
        logger.info("[%d/%d] %s", idx + 1, len(items), qid)

        safe_qid = qid.replace("/", "_").replace(".", "_")
        db_path = str(tmp_dir / f"eval_{safe_qid}.db")
        pipeline: MemoryRetrievalPipeline | None = None
        try:
            await reset_db(db_path)
            pipeline = MemoryRetrievalPipeline(db_path=db_path)

            sessions = entry.get("haystack_sessions", [])
            dates = entry.get("haystack_dates", [])
            sess_ids = entry.get("haystack_session_ids", [])

            for sid, sess, date_str in zip(sess_ids, sessions, dates):
                event_time = _parse_lme_date(date_str)
                for turn_idx, turn in enumerate(sess, start=1):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    provenance = {
                        "lme_session_id": sid,
                        "lme_turn_index": turn_idx,
                        "lme_turn_role": role,
                    }
                    if role == "user":
                        await pipeline.ingest(
                            content, event_time=event_time, role="user",
                            provenance=provenance,
                        )
                    elif role == "assistant":
                        await pipeline.ingest(
                            content, event_time=event_time, role="assistant",
                            provenance=provenance,
                        )

            ref_time = _parse_lme_date(entry.get("question_date")) or _max_haystack_time(dates)
            result = await pipeline.answer(
                entry["question"], reference_time=ref_time, return_evidence=True,
                qid=qid,
            )
            hypothesis = result["answer"] if isinstance(result, dict) else result
            evidence = result.get("evidence", []) if isinstance(result, dict) else []

            await write_result({
                "question_id": qid,
                "hypothesis": hypothesis,
                "evidence": evidence,
            })
            completed += 1

            # 单题调用追踪：读取 answer() 期间经 contextvar 累积的 CallTrace，写入 call_log.jsonl。
            # results 先写（评测真值），call_log 后写；超时/异常的题不会走到这里，
            # last_trace 可能残缺故不写，留待 resume 重跑。
            trace_rec: dict = {"question_id": qid}
            if pipeline.last_trace is not None:
                trace_rec.update(pipeline.last_trace.to_dict())
            await write_call_log(trace_rec)

            logger.info("  [%d/%d] Q: %s", idx + 1, len(items), entry["question"][:60])
            logger.info("  [%d/%d] A: %s", idx + 1, len(items), str(hypothesis)[:60])
        finally:
            # 释放 DB 连接 + 清理临时文件
            try:
                if pipeline is not None:
                    await pipeline.container.db.close()
            except Exception as e:
                logger.warning("close db for %s failed: %s", qid, e)
            _db_map.pop(db_path, None)
            try:
                Path(db_path).unlink(missing_ok=True)
            except Exception:
                pass

    async def run_with_sem(entry: dict, idx: int) -> None:
        qid = entry["question_id"]
        async with sem:
            try:
                await asyncio.wait_for(
                    process_one(entry, idx),
                    timeout=args.per_item_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("[%d/%d] %s TIMEOUT after %ds", idx + 1, len(items),
                             qid, args.per_item_timeout)
                await write_error({"question_id": qid, "error": "timeout",
                                   "timeout_s": args.per_item_timeout})
            except Exception as e:
                logger.exception("[%d/%d] %s FAILED: %s", idx + 1, len(items), qid, e)
                await write_error({"question_id": qid, "error": type(e).__name__,
                                   "message": str(e)})

    tasks = [run_with_sem(entry, i) for i, entry in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Done. completed=%d, output=%s, errors=%s",
                completed, output_path, errors_path)

    # Token 用量汇总
    from internal.util.token_tracker import tracker
    run_label = f"evaluate_longmemeval:{output_path.name}:completed={completed}"
    tracker.write_run_summary(run_label)
    snap = tracker.snapshot()

    # 统计输出
    print("\n=== LongMemEval Results Summary ===")
    print(f"Completed this run: {completed}")
    print(f"Total skipped (resume): {len(items) - len(pending)}")
    print(f"Output file: {output_path}")
    if errors_path.exists():
        print(f"Errors file: {errors_path}")
    print("\n--- Token usage ---")
    for model, u in snap.items():
        print(f"  {model:9s} calls={u['calls']:6d} "
              f"prompt={u['prompt_tokens']:>9d} "
              f"completion={u['completion_tokens']:>9d} "
              f"total={u['total_tokens']:>9d}")
    grand = sum(u["total_tokens"] for u in snap.values())
    print(f"  {'GRAND':9s} {' '*22}{' '*11}{' '*22}total={grand:>9d}")
    print("===================================\n")


if __name__ == "__main__":
    asyncio.run(main())
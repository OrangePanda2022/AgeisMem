"""只跑之前失败的 question_id，对比 P0-P4 改动效果。

用法：
  PYTHONPATH=. uv run python scripts/eval_failed_cases.py \
      --output /home/manjaro/tmp/p1234_test/failed_rerun.jsonl \
      --max 20
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# 复用 evaluate_longmemeval 的解析/编排逻辑
from scripts.evaluate_longmemeval import (
    _resolve_data_path, _parse_lme_date, _max_haystack_time,
    DEFAULT_DATA_DIR, DEFAULT_TMP_DIR, DEFAULT_PER_ITEM_TIMEOUT,
    _load_completed_qids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("failed_cases")


# 之前 full500 跑出的失败 qid（86 个）
FAILED_QIDS = {
    "gpt4_2487a7cb", "982b5123", "gpt4_93159ced", "gpt4_0b2f1d21", "gpt4_b4a80587",
    "dcfa8644", "gpt4_5438fa52", "gpt4_fe651585", "0a995998", "dd2973ad",
    "36b9f61e", "gpt4_15e38248", "2e6d26dc", "gpt4_c27434e8_abs", "eeda8a6d",
    "a9f6b44c", "9d25d4e0", "7024f17c", "6a1eabeb", "edced276",
    "07741c44", "gpt4_2f8be40d", "59524333", "ba61f0b9", "07741c45",
    "031748ae_abs", "afdc33df", "caf03d32", "0ddfec37_abs", "57f827a0",
    "b6025781", "7161e7e2", "38146c39", "1c0ddc50", "75f70248",
    "8752c811", "1a1907b4", "41275add", "cc539528", "d24813b1",
    "1d4e3b97", "e8a79c70", "7a8d0b71", "b0479f84", "70b3e69b",
    "561fabcd", "3b6f954b", "fca70973", "51a45a95", "af8d2e46",
    "58ef2f1c", "0a34ad58", "dccbc061", "577d4d32", "8a137a7f",
    "c14c00dd", "71017276", "gpt4_1916e0ea", "9a707b81", "gpt4_8279ba02",
    "gpt4_4ef30696", "370a8ff4", "gpt4_45189cb4", "gpt4_e061b84f", "5e1b23de",
    "6e984301", "71017277", "gpt4_fa19884d", "gpt4_e414231f", "eac54add",
    "2ebe6c92", "gpt4_7bc6cf22", "60036106", "6e984302", "51c32626",
    "4bc144e2", "9ee3ecd6", "27016adc", "gpt4_1e4a8aec", "92a0aa75",
    "8979f9ec", "f35224e0", "157a136e", "c18a7dc8", "09ba9854_abs",
    "a96c20ee_abs",
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="longmemeval_oracle.json")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", default="/home/manjaro/tmp/p1234_test/failed_rerun.jsonl")
    parser.add_argument("--errors", default=None)
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--max", type=int, default=None, help="最多跑多少失败案例")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--per-item-timeout", type=int, default=DEFAULT_PER_ITEM_TIMEOUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data, args.data_dir)
    from main import MemoryRetrievalPipeline
    from internal.infra.database.sqlite import reset_db, _db_map

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 筛出失败案例
    failed_items = [e for e in data if e.get("question_id") in FAILED_QIDS]
    if args.max:
        failed_items = failed_items[:args.max]
    logger.info("Will rerun %d previously failed cases", len(failed_items))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = Path(args.errors) if args.errors else output_path.with_name("errors.jsonl")
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.no_resume and output_path.exists():
        output_path.unlink()
    done_qids = _load_completed_qids(str(output_path))
    pending = [(i, e) for i, e in enumerate(failed_items) if e.get("question_id") not in done_qids]
    logger.info("Pending: %d (skipped %d already done)", len(pending), len(failed_items) - len(pending))

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    error_lock = asyncio.Lock()
    completed = 0

    async def write_result(record: dict) -> None:
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def write_error(record: dict) -> None:
        async with error_lock:
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def process_one(entry: dict, idx: int) -> None:
        nonlocal completed
        qid = entry["question_id"]
        logger.info("[%d/%d] %s", idx + 1, len(failed_items), qid)

        safe_qid = qid.replace("/", "_").replace(".", "_")
        db_path = str(tmp_dir / f"eval_failed_{safe_qid}.db")
        pipeline = None
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
                    await pipeline.ingest(
                        content, event_time=event_time, role=role,
                        provenance=provenance,
                    )

            ref_time = _parse_lme_date(entry.get("question_date")) or _max_haystack_time(dates)
            result = await pipeline.answer(
                entry["question"], reference_time=ref_time, return_evidence=True,
            )
            hypothesis = result["answer"] if isinstance(result, dict) else result
            evidence = result.get("evidence", []) if isinstance(result, dict) else []

            await write_result({
                "question_id": qid,
                "hypothesis": hypothesis,
                "evidence": evidence,
            })
            completed += 1
            logger.info("  [%d/%d] %s done: %s",
                        idx + 1, len(failed_items), qid, str(hypothesis)[:80])
        finally:
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
                logger.error("[%d/%d] %s TIMEOUT", idx + 1, len(failed_items), qid)
                await write_error({"question_id": qid, "error": "timeout"})
            except Exception as e:
                logger.exception("[%d/%d] %s FAILED: %s", idx + 1, len(failed_items), qid, e)
                await write_error({"question_id": qid, "error": type(e).__name__, "message": str(e)})

    tasks = [run_with_sem(entry, i) for i, entry in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Done. completed=%d, output=%s", completed, output_path)
    print(f"\n=== Failed cases rerun summary ===")
    print(f"Completed: {completed}")
    print(f"Output: {output_path}")
    if errors_path.exists():
        print(f"Errors: {errors_path}")


if __name__ == "__main__":
    asyncio.run(main())

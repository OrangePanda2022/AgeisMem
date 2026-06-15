"""Debug 单题运行器：跑一题 LongMemEval，dump 完整召回管线日志到 JSON。

用法：
  PYTHONPATH=. uv run python scripts/debug_one_question.py \\
    --qid 0bb5a684 \\
    --data longmemeval_oracle.json \\
    --out /home/manjaro/tmp/debug_0bb5a684.json

支持 --qid 部分匹配（前缀匹配第一个）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("debug_one")

DEFAULT_DATA_DIR = "/home/manjaro/AI/LongMemEval/data"
DEFAULT_TMP_DIR = "/home/manjaro/tmp"


def _resolve_data(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return p
    candidate = Path(DEFAULT_DATA_DIR) / data_arg
    if candidate.exists():
        return candidate
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"data not found: {data_arg}")


def _max_haystack_time(dates: list[str]) -> datetime | None:
    parsed: list[datetime] = []
    for d in dates or []:
        if not d:
            continue
        try:
            parsed.append(datetime.fromisoformat(d))
        except Exception:
            continue
    return max(parsed) if parsed else None


async def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval single-question debug")
    parser.add_argument("--qid", required=True, help="question_id (full or prefix)")
    parser.add_argument("--data", default="longmemeval_oracle.json")
    parser.add_argument("--out", default=None,
                        help="debug JSON output path; default /home/manjaro/tmp/debug_<qid>.json")
    args = parser.parse_args()

    data_path = _resolve_data(args.data)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entry = next(
        (e for e in data if e["question_id"] == args.qid
         or e["question_id"].startswith(args.qid)),
        None,
    )
    if entry is None:
        logger.error("qid %r not found in %s", args.qid, data_path)
        sys.exit(1)
    qid = entry["question_id"]
    logger.info("matched qid=%s question_type=%s", qid, entry.get("question_type"))

    out_path = Path(args.out) if args.out else Path(DEFAULT_TMP_DIR) / f"debug_{qid.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from main import MemoryRetrievalPipeline
    from internal.infra.database.sqlite import reset_db, _db_map

    safe_qid = qid.replace("/", "_").replace(".", "_")
    db_path = str(Path(DEFAULT_TMP_DIR) / f"debug_{safe_qid}.db")
    await reset_db(db_path)
    pipeline = MemoryRetrievalPipeline(db_path=db_path)

    sessions = entry.get("haystack_sessions", [])
    dates = entry.get("haystack_dates", [])
    sess_ids = entry.get("haystack_session_ids", [])

    logger.info("ingesting %d sessions ...", len(sessions))
    for sid, sess, date_str in zip(sess_ids, sessions, dates):
        try:
            event_time = datetime.fromisoformat(date_str) if date_str else None
        except Exception:
            event_time = None
        for turn in sess:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                await pipeline.ingest(content, event_time=event_time, role="user")
            elif role == "assistant":
                await pipeline.ingest(content, event_time=event_time, role="assistant")

    ref_time = _max_haystack_time(dates)
    logger.info("ingest done. answering with debug → %s", out_path)
    hypothesis = await pipeline.answer(
        entry["question"],
        reference_time=ref_time,
        debug_path=str(out_path),
        qid=qid,
    )

    print(json.dumps({
        "question_id": qid,
        "question": entry["question"],
        "answer_gt": entry.get("answer"),
        "hypothesis": hypothesis,
        "debug_path": str(out_path),
    }, ensure_ascii=False, indent=2))

    try:
        await pipeline.container.db.close()
    except Exception:
        pass
    _db_map.pop(db_path, None)


if __name__ == "__main__":
    asyncio.run(main())

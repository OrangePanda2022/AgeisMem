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
import re
import sys
from datetime import datetime, timezone
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


def _max_haystack_time(dates: list[str]) -> datetime | None:
    parsed: list[datetime] = []
    for d in dates or []:
        dt = _parse_lme_date(d)
        if dt is not None:
            parsed.append(dt)
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
        event_time = _parse_lme_date(date_str)
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

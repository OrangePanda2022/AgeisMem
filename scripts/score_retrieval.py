"""LongMemEval 检索层评测脚本（对齐官方 run_retrieval.py + print_retrieval_metrics.py）。

复刻官方 process_item_flat_index 的 corpus 构建（仅 user turn，id=`<sess_id>_<i_turn+1>`，
has_answer=False 时 'answer' → 'noans'），调 evaluate_retrieval / evaluate_retrieval_turn2session
计算 recall_any/recall_all/ndcg_any @k（k∈{1,3,5,10,30,50}）的 session/turn 两层指标。

rankings 来源：AegisMem `answer(return_evidence=True)` 返回的 evidence，按 rrf_score 降序、
映射到 turn id 后去重、再转成 corpus index。

用法：
  PYTHONPATH=. uv run python scripts/score_retrieval.py \\
      --hyp /home/manjaro/tmp/lme_oracle.jsonl \\
      --ref /home/manjaro/AI/LongMemEval/data/longmemeval_oracle.json

支持断点续跑（--resume）与 errors.jsonl 错误隔离。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# 让同目录下的 lme_eval_utils 可直接 import（scripts 目录非包时也工作）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme_eval_utils import (  # noqa: E402
    evaluate_retrieval,
    evaluate_retrieval_turn2session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("score_retrieval")


KS = [1, 3, 5, 10, 30, 50]


def _load_jsonl_or_json(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    text_strip = text.lstrip()
    if text_strip.startswith("["):
        return json.loads(text)
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def build_corpus_turn(haystack_session_ids: list[str],
                      haystack_sessions: list[list[dict]]) -> tuple[list[str], list[str]]:
    """复刻官方 process_item_flat_index 的 turn 粒度逻辑。

    返回 (corpus, corpus_ids)：
      - corpus_ids 仅含 user turn
      - id 格式 `<sess_id>_<i_turn+1>`，i_turn 为 enumerate(sess, start=1) 索引
        （含所有 role 的 enumerate，对齐官方）
      - 若 user turn 的 has_answer=False 且 sess_id 含 'answer'，把 'answer'
        替换成 'noans'（同官方）
    """
    corpus: list[str] = []
    corpus_ids: list[str] = []
    for sess_id, sess in zip(haystack_session_ids, haystack_sessions):
        for i_turn, turn in enumerate(sess, start=1):
            if turn.get("role") != "user":
                continue
            content = turn.get("content", "")
            tid = f"{sess_id}_{i_turn}"
            if "answer" in sess_id and not turn.get("has_answer", False):
                tid = tid.replace("answer", "noans")
            corpus.append(content)
            corpus_ids.append(tid)
    return corpus, corpus_ids


def evidence_to_rankings(evidence: list[dict],
                         corpus_id_to_idx: dict[str, int]) -> list[int]:
    """evidence → rankings（corpus index 数组，按 rrf_score 降序去重）。

    每条 fact → turn id `<source_session_id>_<source_turn_index>`；映射不到
    corpus（如来自 assistant turn 或缺 provenance）的 fact 被丢弃。去重时
    保留首次出现（即 rrf_score 最高的）。
    """
    ranked = sorted(
        evidence,
        key=lambda x: x.get("rrf_score", 0.0),
        reverse=True,
    )
    seen: set[str] = set()
    rankings: list[int] = []
    for ev in ranked:
        sid = ev.get("source_session_id")
        tidx = ev.get("source_turn_index")
        if sid is None or tidx is None:
            continue
        tid = f"{sid}_{tidx}"
        if tid in seen:
            continue
        seen.add(tid)
        idx = corpus_id_to_idx.get(tid)
        if idx is None:
            continue
        rankings.append(idx)
    return rankings


def evaluate_one(hyp_entry: dict, ref_entry: dict) -> dict:
    """对单题计算 retrieval_results.metrics.{session, turn}。

    返回 dict：{
        "question_id": ..., "question_type": ...,
        "retrieval_results": {
            "ranked_turn_ids": [...],
            "metrics": {"session": {...}, "turn": {...}},
        }
    }
    """
    haystack_session_ids = ref_entry.get("haystack_session_ids", [])
    haystack_sessions = ref_entry.get("haystack_sessions", [])
    corpus, corpus_ids = build_corpus_turn(haystack_session_ids, haystack_sessions)
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    correct_docs = list({cid for cid in corpus_ids if "answer" in cid})

    evidence = hyp_entry.get("evidence", [])
    rankings = evidence_to_rankings(evidence, corpus_id_to_idx)

    turn_metrics: dict[str, float] = {}
    session_metrics: dict[str, float] = {}
    for k in KS:
        r_any, r_all, ndcg = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
        turn_metrics[f"recall_any@{k}"] = r_any
        turn_metrics[f"recall_all@{k}"] = r_all
        turn_metrics[f"ndcg_any@{k}"] = ndcg
        s_any, s_all, s_ndcg = evaluate_retrieval_turn2session(
            rankings, correct_docs, corpus_ids, k=k,
        )
        session_metrics[f"recall_any@{k}"] = s_any
        session_metrics[f"recall_all@{k}"] = s_all
        session_metrics[f"ndcg_any@{k}"] = s_ndcg

    ranked_turn_ids = [corpus_ids[i] for i in rankings]
    return {
        "question_id": hyp_entry.get("question_id"),
        "question_type": ref_entry.get("question_type", "unknown"),
        "retrieval_results": {
            "ranked_turn_ids": ranked_turn_ids,
            "metrics": {
                "session": session_metrics,
                "turn": turn_metrics,
            },
        },
    }


def has_target_user_turn(ref_entry: dict) -> bool:
    """官方 run_retrieval.py 在 print 时跳过"没有 user 侧 has_answer=True 的题"。

    这里返回 False 表示该题应跳过统计。
    """
    for sess in ref_entry.get("haystack_sessions", []):
        for turn in sess:
            if turn.get("role") == "user" and turn.get("has_answer") is True:
                return True
    return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval Retrieval Scorer")
    parser.add_argument("--hyp", required=True, help="evaluate_longmemeval 输出 jsonl（含 evidence）")
    parser.add_argument("--ref", default="/home/manjaro/AI/LongMemEval/data/longmemeval_oracle.json",
                        help="LongMemEval 参考 JSON")
    parser.add_argument("--out", default=None,
                        help="检索评分输出 jsonl，默认 <hyp>.retrieval.jsonl")
    parser.add_argument("--errors", default=None,
                        help="错误记录文件，默认 <out> 同目录的 errors.jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="若 --out 已存在，跳过其中已处理的 question_id")
    args = parser.parse_args()

    hyp_list = _load_jsonl_or_json(args.hyp)
    ref_list = _load_jsonl_or_json(args.ref)
    qid2ref = {e["question_id"]: e for e in ref_list}

    out_path = Path(args.out) if args.out else Path(args.hyp).with_suffix(
        Path(args.hyp).suffix + ".retrieval.jsonl"
    )
    errors_path = Path(args.errors) if args.errors else out_path.with_name("errors.jsonl")

    done_qids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in _load_jsonl_or_json(str(out_path)):
            qid = rec.get("question_id")
            if qid:
                done_qids.add(qid)
        logger.info("resume: %d already done, skipping", len(done_qids))
    elif out_path.exists():
        out_path.unlink()

    pending = [h for h in hyp_list if h.get("question_id") not in done_qids]
    logger.info("Scoring %d entries (skipped %d) → %s",
                len(pending), len(hyp_list) - len(pending), out_path)

    write_lock = asyncio.Lock()
    error_lock = asyncio.Lock()
    n_skipped = 0
    n_errors = 0

    async def score_one(entry: dict) -> None:
        nonlocal n_skipped, n_errors
        qid = entry.get("question_id")
        ref = qid2ref.get(qid)
        if ref is None:
            logger.warning("skip %s: not in reference", qid)
            n_skipped += 1
            return
        try:
            result = evaluate_one(entry, ref)
        except Exception as e:
            logger.exception("evaluate %s failed: %s", qid, e)
            n_errors += 1
            async with error_lock:
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question_id": qid,
                        "error": type(e).__name__,
                        "message": str(e),
                    }, ensure_ascii=False) + "\n")
            return
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 纯本地计算，并发只用于并行处理大量题目（CPU-bound 但每个题很快）
    await asyncio.gather(*(score_one(h) for h in pending), return_exceptions=False)

    # 聚合报告（对齐 print_retrieval_metrics.py）
    all_records = _load_jsonl_or_json(str(out_path)) if out_path.exists() else []
    # 跳过 _abs 题（官方逻辑）；跳过无 user-target 题（官方 print 时也跳过）
    kept: list[dict] = []
    ignored_abs: set[str] = set()
    ignored_no_target: set[str] = set()
    for rec in all_records:
        qid = rec.get("question_id", "")
        if "_abs" in qid:
            ignored_abs.add(qid)
            continue
        ref = qid2ref.get(qid)
        if ref is not None and not has_target_user_turn(ref):
            ignored_no_target.add(qid)
            continue
        kept.append(rec)

    logger.info("Ignored %d abstention: %s", len(ignored_abs), sorted(ignored_abs))
    logger.info("Additionally ignored %d no-target: %s",
                len(ignored_no_target), sorted(ignored_no_target))

    sess_metric_names = ["recall_all@5", "ndcg_any@5", "recall_all@10", "ndcg_any@10"]
    turn_metric_names = ["recall_all@5", "ndcg_any@5", "recall_all@10",
                         "ndcg_any@10", "recall_all@50", "ndcg_any@50"]

    def _mean(metric_name: str, level: str) -> float:
        vals = [
            r["retrieval_results"]["metrics"][level].get(metric_name)
            for r in kept
            if r.get("retrieval_results", {}).get("metrics", {}).get(level)
        ]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n=== LongMemEval Retrieval Score (official-aligned) ===")
    print(f"hyp    : {args.hyp}")
    print(f"out    : {out_path}")
    print(f"errors : {errors_path} (n={n_errors})")
    print(f"kept   : {len(kept)} (ignored abstention={len(ignored_abs)}, no_target={len(ignored_no_target)})")
    print("\nSession-level metrics:")
    print("   " + ", ".join(
        f"{name} = {_mean(name, 'session'):.4f}" for name in sess_metric_names
    ))
    print("Turn-level metrics:")
    print("   " + ", ".join(
        f"{name} = {_mean(name, 'turn'):.4f}" for name in turn_metric_names
    ))
    print("==========================================================\n")


if __name__ == "__main__":
    asyncio.run(main())

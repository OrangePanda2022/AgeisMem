"""LongMemEval hypothesis 打分脚本。

复用项目内的 JudgeClient（Anthropic 协议，可指向任意兼容网关）按官方
LongMemEval 的 prompt 模板对 hypothesis 进行 yes/no 判断，输出整体准确率与
按 question_type 分桶的准确率。

用法：
  PYTHONPATH=. uv run python scripts/score_longmemeval.py \\
      --hyp /home/manjaro/tmp/lme_50.jsonl \\
      --ref /home/manjaro/AI/LongMemEval/data/longmemeval_oracle.json

打分前请确保 settings.judge_api_key / judge_base_url / judge_model 已填。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("score_longmemeval")


# 官方 prompt 模板（来自 LongMemEval/src/evaluation/evaluate_qa.py）
_TEMPLATE_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. \n\nQuestion: {q}\n\nCorrect Answer: "
    "{a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still "
    "correct. \n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long "
    "as the updated answer is the required answer.\n\nQuestion: {q}\n\nCorrect "
    "Answer: {a}\n\nModel Response: {r}\n\nIs the model response correct? "
    "Answer yes or no only."
)

_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {q}\n\nRubric: {a}\n\nModel Response: {r}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {q}\n\nExplanation: {a}\n\nModel Response: {r}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def build_prompt(question_type: str, question: str, answer: str, response: str,
                 abstention: bool) -> str:
    if abstention:
        tpl = _TEMPLATE_ABSTENTION
    elif question_type in ("single-session-user", "single-session-assistant", "multi-session"):
        tpl = _TEMPLATE_DEFAULT
    elif question_type == "temporal-reasoning":
        tpl = _TEMPLATE_TEMPORAL
    elif question_type == "knowledge-update":
        tpl = _TEMPLATE_KNOWLEDGE_UPDATE
    elif question_type == "single-session-preference":
        tpl = _TEMPLATE_PREFERENCE
    else:
        # 未知类型走默认模板，避免直接 raise
        logger.warning("unknown question_type=%s, using default template", question_type)
        tpl = _TEMPLATE_DEFAULT
    return tpl.format(q=question, a=answer, r=response)


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


async def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval Scorer")
    parser.add_argument("--hyp", required=True, help="hypothesis jsonl 文件")
    parser.add_argument("--ref", default="/home/manjaro/AI/LongMemEval/data/longmemeval_oracle.json",
                        help="LongMemEval 参考 JSON")
    parser.add_argument("--out", default=None,
                        help="评分结果输出 jsonl，默认 <hyp>.scored.jsonl")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="judge 并发；下游同时受 settings.llm_max_concurrency 限流")
    parser.add_argument("--resume", action="store_true",
                        help="若 --out 已存在，跳过其中已打分的 question_id")
    args = parser.parse_args()

    from internal.config.settings import settings as _settings
    if not _settings.judge_api_key or not _settings.judge_base_url or not _settings.judge_model:
        logger.error(
            "JudgeClient 未配置完整：judge_api_key / judge_base_url / judge_model "
            "需先在 internal/config/settings.py 或 .env 中填好"
        )
        sys.exit(2)

    from internal.infra.models.judge.judge import judge_client

    hyp_list = _load_jsonl_or_json(args.hyp)
    ref_list = _load_jsonl_or_json(args.ref)
    qid2ref = {e["question_id"]: e for e in ref_list}

    out_path = Path(args.out) if args.out else Path(args.hyp).with_suffix(
        Path(args.hyp).suffix + ".scored.jsonl"
    )

    done_qids: set[str] = set()
    if args.resume and out_path.exists():
        for rec in _load_jsonl_or_json(str(out_path)):
            qid = rec.get("question_id")
            if qid:
                done_qids.add(qid)
        logger.info("resume: %d already scored, skipping", len(done_qids))
    elif out_path.exists():
        out_path.unlink()

    pending = [h for h in hyp_list if h.get("question_id") not in done_qids]
    logger.info("Scoring %d hypotheses (skipped %d) → %s",
                len(pending), len(hyp_list) - len(pending), out_path)

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    qtype_correct: dict[str, list[int]] = defaultdict(list)
    n_skipped = 0

    async def score_one(entry: dict) -> None:
        nonlocal n_skipped
        qid = entry["question_id"]
        ref = qid2ref.get(qid)
        if ref is None:
            logger.warning("skip %s: not in reference", qid)
            n_skipped += 1
            return
        qtype = ref.get("question_type", "unknown")
        prompt = build_prompt(
            qtype, ref["question"], ref["answer"], entry.get("hypothesis", ""),
            abstention="_abs" in qid,
        )
        async with sem:
            try:
                resp = await judge_client.judge(prompt)
            except Exception as e:
                logger.error("judge %s failed: %s", qid, e)
                resp = ""
        label = "yes" in resp.lower()
        rec = {
            "question_id": qid,
            "question_type": qtype,
            "hypothesis": entry.get("hypothesis", ""),
            "answer": ref["answer"],
            "judge_response": resp,
            "autoeval_label": label,
        }
        qtype_correct[qtype].append(1 if label else 0)
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(score_one(h) for h in pending), return_exceptions=False)

    # 合并 resume 中已评分的部分到统计
    if done_qids:
        for rec in _load_jsonl_or_json(str(out_path)):
            qid = rec.get("question_id")
            if qid in done_qids:
                qtype_correct[rec.get("question_type", "unknown")].append(
                    1 if rec.get("autoeval_label") else 0
                )

    total = sum(len(v) for v in qtype_correct.values())
    correct = sum(sum(v) for v in qtype_correct.values())
    overall = correct / total if total else 0.0

    print("\n=== LongMemEval Score ===")
    print(f"hyp    : {args.hyp}")
    print(f"out    : {out_path}")
    print(f"scored : {total}  (skipped: {n_skipped})")
    print(f"overall: {overall:.4f}  ({correct}/{total})")
    print("by question_type:")
    for qt in sorted(qtype_correct):
        v = qtype_correct[qt]
        if v:
            print(f"  {qt:30s} {sum(v)/len(v):.4f}  ({sum(v)}/{len(v)})")

    from internal.util.token_tracker import tracker
    run_label = f"score_longmemeval:{Path(args.hyp).name}:scored={total}"
    tracker.write_run_summary(run_label)
    snap = tracker.snapshot()
    print("\n--- Token usage ---")
    for model, u in snap.items():
        print(f"  {model:9s} calls={u['calls']:6d} "
              f"prompt={u['prompt_tokens']:>9d} "
              f"completion={u['completion_tokens']:>9d} "
              f"total={u['total_tokens']:>9d}")
    grand = sum(u["total_tokens"] for u in snap.values())
    print(f"  GRAND total={grand}")
    print("=========================\n")


if __name__ == "__main__":
    asyncio.run(main())

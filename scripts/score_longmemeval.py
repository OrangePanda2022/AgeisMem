"""LongMemEval QA 评测脚本（对齐官方 evaluate_qa.py + print_qa_metrics.py）。

复刻官方 5 个 prompt 模板与 autoeval_label = {"model", "label"} 格式；
走 settings.judge_* 的 OpenAI 兼容网关；保留 asyncio 并发、断点续跑
与 errors.jsonl 错误隔离。

Judge 调用参数对齐官方：temperature=0、max_tokens=10、无 reasoning_effort。

用法：
  PYTHONPATH=. uv run python scripts/score_longmemeval.py \\
      --hyp /home/manjaro/tmp/lme_oracle.jsonl \\
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


# 官方 prompt 模板（逐字复刻自 LongMemEval/src/evaluation/evaluate_qa.py）
_TEMPLATE_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
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
    "correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_TEMPLATE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long "
    "as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect "
    "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)

# 官方 preference 模板原文（注意跟旧脚本的改写版不同——rubric 语义、
# "does not need to reflect all the points"）
_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)

_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
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
        logger.warning("unknown question_type=%s, using default template", question_type)
        tpl = _TEMPLATE_DEFAULT
    return tpl.format(question, answer, response)


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
    parser = argparse.ArgumentParser(description="LongMemEval QA Scorer (official-aligned)")
    parser.add_argument("--hyp", required=True, help="hypothesis jsonl 文件")
    parser.add_argument("--ref", default="/home/manjaro/AI/LongMemEval/data/longmemeval_oracle.json",
                        help="LongMemEval 参考 JSON")
    parser.add_argument("--out", default=None,
                        help="评分结果输出 jsonl，默认 <hyp>.scored.jsonl")
    parser.add_argument("--errors", default=None,
                        help="错误记录文件，默认 <out> 同目录的 errors.jsonl")
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
    errors_path = Path(args.errors) if args.errors else out_path.with_name("errors.jsonl")

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
    error_lock = asyncio.Lock()
    n_skipped = 0
    n_errors = 0

    async def score_one(entry: dict) -> None:
        nonlocal n_skipped, n_errors
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
                # mimo-v2.5-pro 等 reasoning 模型需要先推理再给 yes/no。
                # max_tokens=10 会截断在 reasoning 阶段，content 为空，
                # judge.py 兜底返回 reasoning_content 开头（如 "First, the question is..."），
                # 导致 "yes" in resp 误判 False。
                # 修：max_tokens=2048 + reasoning_effort="low"，让模型完整输出 yes/no，
                # 再从末尾抽取最终的 yes/no 判定。
                resp = await judge_client.judge(prompt, max_tokens=2048)
            except Exception as e:
                logger.error("judge %s failed: %s", qid, e)
                n_errors += 1
                async with error_lock:
                    with errors_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "question_id": qid, "error": type(e).__name__,
                            "message": str(e),
                        }, ensure_ascii=False) + "\n")
                return
        # P4 验证修：reasoning 模型输出形如 "First, the question is... reasoning... Yes."
        # 从末尾抽最后的 yes/no 判定，避免 reasoning 文本里的 "yes"/"no" 干扰。
        resp_lower = resp.lower().strip()
        # 优先匹配末尾的 yes/no（可能带标点）
        import re as _re
        m = _re.search(r'\b(yes|no)\b\s*[.!?]*\s*$', resp_lower)
        if m:
            label = m.group(1) == "yes"
        else:
            # 兜底：整段含 yes 且不含 no（保守）
            label = "yes" in resp_lower and "no" not in resp_lower
        rec = {
            "question_id": qid,
            "question_type": qtype,
            "hypothesis": entry.get("hypothesis", ""),
            "answer": ref["answer"],
            "judge_response": resp,
            "autoeval_label": {
                "model": _settings.judge_model,
                "label": label,
            },
        }
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(score_one(h) for h in pending), return_exceptions=False)

    # 合并 resume 中已评分的部分到统计
    all_records = _load_jsonl_or_json(str(out_path)) if out_path.exists() else []
    type2acc: dict[str, list[int]] = defaultdict(list)
    abstention_acc: list[int] = []
    for rec in all_records:
        label_obj = rec.get("autoeval_label")
        label = (
            label_obj.get("label")
            if isinstance(label_obj, dict)
            else bool(label_obj)
        )
        qtype = rec.get("question_type", "unknown")
        type2acc[qtype].append(1 if label else 0)
        if "_abs" in rec.get("question_id", ""):
            abstention_acc.append(1 if label else 0)

    # 6 类 task（与 print_qa_metrics.py 完全一致）
    canonical_types = [
        "single-session-user", "single-session-preference", "single-session-assistant",
        "multi-session", "temporal-reasoning", "knowledge-update",
    ]
    type2acc = {t: type2acc.get(t, []) for t in canonical_types}

    all_acc = [x for v in type2acc.values() for x in v]
    overall = (sum(all_acc) / len(all_acc)) if all_acc else 0.0
    task_means = [sum(v) / len(v) for v in type2acc.values() if v]
    task_averaged = sum(task_means) / len(task_means) if task_means else 0.0
    abst_acc = (sum(abstention_acc) / len(abstention_acc)) if abstention_acc else 0.0

    print("\n=== LongMemEval QA Score (official-aligned) ===")
    print(f"hyp    : {args.hyp}")
    print(f"out    : {out_path}")
    print(f"errors : {errors_path} (n={n_errors})")
    print(f"skipped: {n_skipped}")
    print(f"scored : {len(all_acc)}")
    print("\nEvaluation results by task:")
    for k in canonical_types:
        v = type2acc[k]
        if v:
            print(f"\t{k}: {sum(v)/len(v):.4f} ({len(v)})")
    print(f"\nTask-averaged Accuracy: {task_averaged:.4f}")
    print(f"Overall Accuracy: {overall:.4f}")
    print(f"Abstention Accuracy: {abst_acc:.4f} ({len(abstention_acc)})")
    print("=================================================\n")

    from internal.util.token_tracker import tracker
    run_label = f"score_longmemeval:{Path(args.hyp).name}:scored={len(all_acc)}"
    tracker.write_run_summary(run_label)
    snap = tracker.snapshot()
    print("--- Token usage ---")
    for model, u in snap.items():
        print(f"  {model:9s} calls={u['calls']:6d} "
              f"prompt={u['prompt_tokens']:>9d} "
              f"completion={u['completion_tokens']:>9d} "
              f"total={u['total_tokens']:>9d}")
    grand = sum(u["total_tokens"] for u in snap.values())
    print(f"  GRAND total={grand}")


if __name__ == "__main__":
    asyncio.run(main())

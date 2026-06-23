"""从 scored.jsonl 生成聚合准确率报告 accuracy_report.json。

输出结构:
{
  "summary": { overall, task_averaged, by_task: {...}, by_label: {correct,wrong}, ... },
  "questions": [
    {
      "question_id": "...",
      "question_type": "...",
      "hypothesis": "LLM 最终回答",
      "answer": "标准答案",
      "correct": true/false,          # autoeval_label.label
      "judge_response": "yes/no/..."  # judge 原始回复
    },
    ...
  ]
}

用法:
  cd /home/cachy/AI/AegisMem
  uv run python scripts/build_accuracy_report.py \
      --scored ~/tmp/results.jsonl.scored.jsonl \
      --out ~/tmp/accuracy_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="score_longmemeval.py 的输出 .scored.jsonl")
    ap.add_argument("--out", required=True, help="输出 accuracy_report.json 路径")
    args = ap.parse_args()

    scored = load_jsonl(Path(args.scored))
    print(f"读入 {len(scored)} 条 scored 记录", file=sys.stderr)

    # 官方 6 类 task 顺序
    canonical_types = [
        "single-session-user", "single-session-preference", "single-session-assistant",
        "multi-session", "temporal-reasoning", "knowledge-update",
    ]

    type2acc: dict[str, list[int]] = defaultdict(list)
    abstention_acc: list[int] = []
    questions = []
    n_correct = 0
    n_wrong = 0
    n_unknown = 0  # judge 失败/label=None

    for rec in scored:
        qid = rec.get("question_id", "")
        qtype = rec.get("question_type", "unknown")
        hyp = rec.get("hypothesis", "")
        ans = rec.get("answer", "")
        label_obj = rec.get("autoeval_label")
        if isinstance(label_obj, dict):
            label = label_obj.get("label")
        else:
            label = bool(label_obj)
        judge_raw = rec.get("judge_response", "")

        # None 表示 judge 失败,未判定
        if label is True:
            correct = True
            n_correct += 1
            type2acc[qtype].append(1)
            if "_abs" in qid:
                abstention_acc.append(1)
        elif label is False:
            correct = False
            n_wrong += 1
            type2acc[qtype].append(0)
            if "_abs" in qid:
                abstention_acc.append(0)
        else:
            correct = None  # 未判定
            n_unknown += 1

        questions.append({
            "question_id": qid,
            "question_type": qtype,
            "hypothesis": hyp,
            "answer": ans,
            "correct": correct,
            "judge_response": judge_raw,
        })

    # 汇总
    all_acc = [x for v in type2acc.values() for x in v]
    overall = (sum(all_acc) / len(all_acc)) if all_acc else 0.0
    task_means = [sum(v) / len(v) for v in type2acc.values() if v]
    task_averaged = sum(task_means) / len(task_means) if task_means else 0.0
    abst_acc = (sum(abstention_acc) / len(abstention_acc)) if abstention_acc else 0.0

    by_task = {}
    for t in canonical_types:
        v = type2acc.get(t, [])
        if v:
            by_task[t] = {
                "accuracy": round(sum(v) / len(v), 4),
                "correct": sum(v),
                "total": len(v),
            }
        else:
            by_task[t] = {"accuracy": None, "correct": 0, "total": 0}

    summary = {
        "total_scored": len(scored),
        "total_judged": n_correct + n_wrong,
        "correct": n_correct,
        "wrong": n_wrong,
        "not_judged": n_unknown,
        "overall_accuracy": round(overall, 4),
        "task_averaged_accuracy": round(task_averaged, 4),
        "abstention_accuracy": round(abst_acc, 4),
        "abstention_total": len(abstention_acc),
        "by_task": by_task,
        "judge_model": (scored[0].get("autoeval_label", {}) or {}).get("model") if scored else None,
    }

    report = {"summary": summary, "questions": questions}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"→ 写入 {out_path}", file=sys.stderr)
    print(f"  Overall: {overall:.4f} ({n_correct}/{n_correct+n_wrong})", file=sys.stderr)
    print(f"  未判定: {n_unknown}", file=sys.stderr)


if __name__ == "__main__":
    main()

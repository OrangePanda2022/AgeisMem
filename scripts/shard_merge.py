"""合并双账号 A/B 结果:剔占位,合成最终 results{suffix}.jsonl / call_log{suffix}.jsonl,errors 去重。

A/B 各自 results 文件含真行(带hypothesis)+占位行({"question_id":..,"_placeholder":true})。
合并:取所有真行,按 question_id 去重(A/B shard 互斥,理论上无重叠,但防御性去重)。

非侵入:原 results{suffix}.jsonl 若存在会被覆盖(A/B 真行即真值,合并是权威来源)。

用法:
  cd /home/cachy/AI/AegisMem
  PYTHONPATH=. uv run python scripts/shard_merge.py                                # oracle
  PYTHONPATH=. uv run python scripts/shard_merge.py --data longmemeval_m_cleaned.json --suffix _2
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

TMP = Path.home() / "tmp"
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO.parent / "LongMemEval" / "data"


def resolve_data(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return p
    cand = DATA_DIR / data_arg
    if cand.exists():
        return cand
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"Data file not found: {data_arg!r} (looked in {DATA_DIR})")


def load_real(path: Path) -> dict[str, dict]:
    """读 jsonl,只留非占位真行,按 question_id 索引(后写覆盖)。"""
    recs: dict[str, dict] = {}
    if not path.exists():
        return recs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [跳过坏行] {path.name}: {line[:60]}")
                continue
            if rec.get("_placeholder"):
                continue
            qid = rec.get("question_id")
            if isinstance(qid, str):
                recs[qid] = rec
    return recs


def merge(a: dict[str, dict], b: dict[str, dict], label: str) -> dict[str, dict]:
    """合并 A/B,冲突时记录警告(A/B shard 互斥应无冲突)。"""
    overlap = set(a) & set(b)
    if overlap:
        print(f"  [警告] {label} A/B 有 {len(overlap)} 个 qid 重叠(取A): {list(overlap)[:3]}")
    out = dict(a)
    out.update(b)  # B 覆盖 A(若有重叠)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="合并双账号 A/B 结果")
    ap.add_argument("--data", default="longmemeval_oracle.json",
                    help="数据集文件(用于最终校验全集 qid,默认 oracle)")
    ap.add_argument("--suffix", default="", help="输出文件后缀(如 _2)")
    args = ap.parse_args()
    sfx = args.suffix

    RES_A = TMP / f"results_A{sfx}.jsonl"
    RES_B = TMP / f"results_B{sfx}.jsonl"
    CL_A = TMP / f"call_log_A{sfx}.jsonl"
    CL_B = TMP / f"call_log_B{sfx}.jsonl"
    OUT_RES = TMP / f"results{sfx}.jsonl"
    OUT_CL = TMP / f"call_log{sfx}.jsonl"
    OUT_ERR = TMP / f"errors{sfx}.jsonl"

    print("=== 合并 results ===")
    a = load_real(RES_A)
    b = load_real(RES_B)
    print(f"  A真行={len(a)} B真行={len(b)}")
    merged = merge(a, b, "results")
    print(f"  合并后真行={len(merged)}")

    with OUT_RES.open("w", encoding="utf-8") as f:
        for qid, rec in merged.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → 写入 {OUT_RES} ({len(merged)} 行)")

    print("\n=== 合并 call_log ===")
    ca = load_real(CL_A)
    cb = load_real(CL_B)
    print(f"  A真行={len(ca)} B真行={len(cb)}")
    merged_cl = merge(ca, cb, "call_log")
    with OUT_CL.open("w", encoding="utf-8") as f:
        for qid, rec in merged_cl.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → 写入 {OUT_CL} ({len(merged_cl)} 行)")

    print("\n=== errors 去重 ===")
    err_recs: dict[str, dict] = {}
    if OUT_ERR.exists():
        with OUT_ERR.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [跳过坏行] errors: {line[:60]}")
                    continue
                qid = rec.get("question_id")
                if isinstance(qid, str):
                    err_recs[qid] = rec
    with OUT_ERR.open("w", encoding="utf-8") as f:
        for qid, rec in err_recs.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  去重后 errors={len(err_recs)} 题")
    c = collections.Counter(r.get("error") for r in err_recs.values())
    print(f"  明细: {dict(c)}")

    # 最终校验
    print("\n=== 最终校验 ===")
    done = set(merged.keys())
    err = set(err_recs.keys())
    print(f"  results 真行={len(done)} errors去重={len(err)}")
    try:
        data_path = resolve_data(args.data)
        import ijson
        all_qids: set[str] = set()
        with data_path.open("rb") as f:
            for obj in ijson.items(f, "item"):
                all_qids.add(obj["question_id"])
    except FileNotFoundError as e:
        print(f"  [警告] {e}(跳过全集校验)")
        all_qids = done | err
    missing = all_qids - done - err
    print(f"  done ∪ err = {len(done | err)}/{len(all_qids)} (应为 {len(all_qids)})")
    print(f"  既未完成也未报错的 qid: {len(missing)} {list(missing)[:5] if missing else ''}")


if __name__ == "__main__":
    main()

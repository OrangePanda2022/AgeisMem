"""双账号分片脚本:把数据集 QID 稳定对半分给两个进程,各自预填对方占位。

非侵入:只读原 results{suffix}.jsonl(作真值源),生成全新的 results_A/B{suffix}.jsonl。
合并时按 _placeholder 标记剔除占位行。

分片规则:qid 稳定哈希 mod 2 → shard0(进程A) / shard1(进程B)。
  - 已完成的 qid(在原 results{suffix}.jsonl 里有真行):真行原样落到所属 shard 文件
  - 未完成的 qid:本 shard 待跑,给对方 shard 预填占位 {"question_id":..,"_placeholder":true}
这样每个进程 resume 时:对方 shard 的题被占位跳过,自己 shard 的未完成题才跑。

全新数据集轮次用 --data 指定数据文件、--suffix 指定输出后缀(如 _2)区分产物。
全新轮次若 results{suffix}.jsonl 不存在 → done 集为空 → 全部待跑,各跑一半。

用法:
  cd /home/cachy/AI/AegisMem
  PYTHONPATH=. uv run python scripts/shard_split.py                                          # oracle,无后缀
  PYTHONPATH=. uv run python scripts/shard_split.py --data longmemeval_m_cleaned.json --suffix _2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO.parent / "LongMemEval" / "data"
TMP = Path.home() / "tmp"


def shard_of(qid: str) -> int:
    """稳定分片:qid md5 mod 2 → 0 或 1。确定性,重跑一致,负载均衡。"""
    h = hashlib.md5(qid.encode("utf-8")).hexdigest()
    return int(h, 16) % 2


def load_jsonl_records(path: Path) -> dict[str, dict]:
    """读 jsonl,按 question_id 索引(后写覆盖前写,去重)。"""
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
                continue
            qid = rec.get("question_id")
            if isinstance(qid, str):
                recs[qid] = rec
    return recs


def resolve_data(data_arg: str) -> Path:
    """支持绝对路径 / 相对路径 / 仅文件名(在 DATA_DIR 下找)。"""
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return p
    cand = DATA_DIR / data_arg
    if cand.exists():
        return cand
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"Data file not found: {data_arg!r} (looked in {DATA_DIR})")


def main() -> None:
    ap = argparse.ArgumentParser(description="双账号分片:qid md5 mod2 对半分")
    ap.add_argument("--data", default="longmemeval_oracle.json",
                    help="LongMemEval 数据文件名或路径(默认 oracle)")
    ap.add_argument("--suffix", default="",
                    help="输出文件后缀,区分不同轮次(如 _2);默认无后缀")
    args = ap.parse_args()
    sfx = args.suffix

    try:
        data_path = resolve_data(args.data)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    ORIG_RESULTS = TMP / f"results{sfx}.jsonl"
    ORIG_CALLLOG = TMP / f"call_log{sfx}.jsonl"
    RES_A = TMP / f"results_A{sfx}.jsonl"
    RES_B = TMP / f"results_B{sfx}.jsonl"
    CL_A = TMP / f"call_log_A{sfx}.jsonl"
    CL_B = TMP / f"call_log_B{sfx}.jsonl"

    # 1) 全集 qid 有序列表(流式:s_cleaned/m_cleaned 大文件避免 OOM)
    import ijson
    all_qids: list[str] = []
    with data_path.open("rb") as f:
        for obj in ijson.items(f, "item"):
            all_qids.append(obj["question_id"])
    all_set = set(all_qids)
    print(f"[全集] {data_path.name}: {len(all_qids)} 题,唯一 qid {len(all_set)} 个")

    # 2) 已完成真行(results + call_log 各读一份)
    orig_res = load_jsonl_records(ORIG_RESULTS)
    orig_cl = load_jsonl_records(ORIG_CALLLOG)
    done_qids = set(orig_res.keys())
    print(f"[已完成] results{sfx}.jsonl 真行 {len(orig_res)} 题")
    print(f"[已完成] call_log{sfx}.jsonl 行 {len(orig_cl)} 题")
    print(f"[已完成] 二者并集 {len(done_qids)} 题")
    unknown = done_qids - all_set
    if unknown:
        print(f"[警告] {len(unknown)} 个已完成 qid 不在全集中(忽略): {list(unknown)[:3]}")
        done_qids &= all_set

    # 3) 分片统计
    shard_qids: dict[int, set[str]] = {0: set(), 1: set()}
    for q in all_qids:
        shard_qids[shard_of(q)].add(q)
    print(f"[分片] shard0(A)={len(shard_qids[0])} shard1(B)={len(shard_qids[1])}")

    done_per_shard = {0: 0, 1: 0}
    for q in done_qids:
        done_per_shard[shard_of(q)] += 1
    print(f"[已完成分布] shard0={done_per_shard[0]} shard1={done_per_shard[1]} "
          f"(各自待跑 {len(shard_qids[0])-done_per_shard[0]} / {len(shard_qids[1])-done_per_shard[1]})")

    # 4) 生成 A/B results 文件
    for shard, res_path, cl_path, label in [
        (0, RES_A, CL_A, "A"),
        (1, RES_B, CL_B, "B"),
    ]:
        own = shard_qids[shard]
        other = shard_qids[1 - shard]
        with res_path.open("w", encoding="utf-8") as fr, cl_path.open("w", encoding="utf-8") as fc:
            res_lines = cl_lines = 0
            own_done = 0
            ph_lines = 0
            # 本 shard 已完成的 qid:写真行(原样)
            for q in all_qids:
                if q in own and q in orig_res:
                    fr.write(json.dumps(orig_res[q], ensure_ascii=False) + "\n")
                    res_lines += 1
                    own_done += 1
            for q in all_qids:
                if q in own and q in orig_cl:
                    fc.write(json.dumps(orig_cl[q], ensure_ascii=False) + "\n")
                    cl_lines += 1
            # 对方 shard 全部题:占位(让本进程跳过)
            for q in all_qids:
                if q in other:
                    fr.write(json.dumps({"question_id": q, "_placeholder": True}, ensure_ascii=False) + "\n")
                    ph_lines += 1
            print(f"[进程{label}] results 真行={res_lines}(本shard已完成) 占位={ph_lines}(对方shard) | "
                  f"call_log 真行={cl_lines} | 待跑={len(own)-own_done}")

    # 5) 校验
    a_recs = load_jsonl_records(RES_A)
    b_recs = load_jsonl_records(RES_B)
    a_real = {q for q, r in a_recs.items() if not r.get("_placeholder")}
    a_ph = {q for q, r in a_recs.items() if r.get("_placeholder")}
    b_real = {q for q, r in b_recs.items() if not r.get("_placeholder")}
    b_ph = {q for q, r in b_recs.items() if r.get("_placeholder")}
    print("\n=== 校验 ===")
    print(f"原 results{sfx}.jsonl 真行(只读未改): {len(orig_res)}")
    print(f"A 真行={len(a_real)} 占位={len(a_ph)} 合计={len(a_recs)}")
    print(f"B 真行={len(b_real)} 占位={len(b_ph)} 合计={len(b_recs)}")
    assert a_real | b_real == done_qids, "真行并集 != 已完成集!"
    assert not (a_real & b_real), "A/B 真行有重复!"
    assert a_ph == shard_qids[1], "A 占位 != shard1 全集!"
    assert b_ph == shard_qids[0], "B 占位 != shard0 全集!"
    print(f"✅ 校验通过:A/B 真行无重复且并集=已完成{len(done_qids)}题;占位=对方shard全集")
    print(f"✅ 进程A待跑 {len(shard_qids[0])-done_per_shard[0]} 题, 进程B待跑 {len(shard_qids[1])-done_per_shard[1]} 题")


if __name__ == "__main__":
    main()

"""构建 5 题 P1-B/C 诊断 Dashboard（自包含 HTML，全中文 UI）。

输入：
  --debug-dir  P1-B/C debug JSON 目录
  --scored     P1-B/C scored JSONL
  --p1a-scored P1-A scored JSONL（用于对比）
  --out         输出 HTML 路径
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

STAGE_ORDER = [
    "forgetting", "query_embed", "draft_recall", "keywords",
    "entity_recall", "fact_recall_per_path", "fact_recall_fused",
    "graph_walk", "contradiction", "mas_scored", "expected_entities",
    "reverse_entity_expansion", "sufficiency_check",
    "expand_keywords_via_graph", "iterative_recall", "cba_budget",
    "final_context", "llm_answer",
]

STAGE_CN = {
    "forgetting": "遗忘衰减",
    "query_embed": "查询嵌入",
    "draft_recall": "草稿召回",
    "keywords": "关键词",
    "entity_recall": "实体召回",
    "fact_recall_per_path": "各路径召回",
    "fact_recall_fused": "融合召回",
    "graph_walk": "图遍历",
    "contradiction": "矛盾检测",
    "mas_scored": "MAS 评分",
    "expected_entities": "预期实体",
    "reverse_entity_expansion": "反向实体扩展",
    "sufficiency_check": "充分性检查",
    "expand_keywords_via_graph": "图扩展关键词",
    "iterative_recall": "迭代召回",
    "cba_budget": "CBA 预算分配",
    "final_context": "最终上下文",
    "llm_answer": "LLM 答案",
}

ANOMALY_RULES = [
    ("🔴 LLM 兜底", "red", lambda t: is_fallback(t)),
    ("🔴 空上下文", "red", lambda t: get_ctx_count(t) == 0),
    ("🟡 迭代召回为空", "yellow", lambda t: iter_empty(t)),
    ("🟡 缺失实体", "yellow", lambda t: has_missing_entities(t)),
    ("🟡 top MAS 过低", "yellow", lambda t: top_mas_low(t, 0.55)),
]


def get_stage(trace: dict, name: str) -> dict | None:
    return (trace.get("stages") or {}).get(name)


def is_fallback(trace: dict) -> bool:
    llm = get_stage(trace, "llm_answer") or {}
    if llm.get("fallback"):
        return True
    parsed = llm.get("parsed") or {}
    answer = parsed.get("answer", "")
    return answer.startswith("基于记忆：")


def get_ctx_count(trace: dict) -> int:
    ctx = get_stage(trace, "final_context") or {}
    return ctx.get("fact_count") or 0


def iter_empty(trace: dict) -> bool:
    it = get_stage(trace, "iterative_recall") or {}
    rounds = it.get("rounds") or []
    if not rounds:
        return False
    return all((r or {}).get("new_facts_count", 0) == 0 for r in rounds)


def has_missing_entities(trace: dict) -> bool:
    rev = get_stage(trace, "reverse_entity_expansion") or {}
    return bool(rev.get("missing_entities"))


def top_mas_low(trace: dict, threshold: float) -> bool:
    mas = get_stage(trace, "mas_scored") or {}
    facts = mas.get("facts") or mas.get("scored_facts") or []
    if not facts:
        return False
    first = facts[0]
    if isinstance(first, dict):
        m = first.get("mas") or first.get("score") or 0
        return m < threshold
    return False


def detect_anomalies(trace: dict) -> list[tuple[str, str]]:
    out = []
    for label, _, check in ANOMALY_RULES:
        try:
            if check(trace):
                out.append(label)
        except Exception:
            pass
    return out


def load_oracle() -> dict[str, dict]:
    p = Path("/home/manjaro/AI/LongMemEval/data/longmemeval_oracle.json")
    with p.open() as f:
        data = json.load(f)
    return {e["question_id"][:8]: e for e in data}


def load_scored(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["question_id"][:8]] = r
    return out


def build_mas_table(payload: dict) -> str:
    facts = payload.get("facts") or payload.get("scored_facts") or []
    if not facts:
        return '<p class="empty">无 MAS 评分数据</p>'
    rows = []
    for i, f in enumerate(facts[:10]):
        if not isinstance(f, dict):
            continue
        m = f.get("mas") or f.get("score") or 0
        c = f.get("content") or ""
        meta = f.get("metadata") or {}
        pref = meta.get("Preference", "") if isinstance(meta, dict) else ""
        rows.append(
            f'<tr><td>{i}</td><td>{m:.3f}</td><td class="content">{html.escape(c[:250])}</td>'
            f'<td>{html.escape(str(pref))}</td></tr>'
        )
    return (
        '<table class="mas-table"><thead><tr>'
        '<th>序号</th><th>MAS</th><th>fact 内容</th><th>偏好</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def build_final_context(payload: dict) -> str:
    count = payload.get("fact_count", 0)
    tokens = payload.get("approx_tokens", 0)
    char_len = payload.get("char_len", 0)
    ctx_str = payload.get("context", "")
    summary = (
        f'<div class="ctx-summary">'
        f'<span>facts: <b>{count}</b></span>'
        f'<span>tokens: <b>{tokens}</b></span>'
        f'<span>chars: <b>{char_len}</b></span>'
        f'</div>'
    )
    body = html.escape(ctx_str[:6000])
    if len(ctx_str) > 6000:
        body += '\n\n[...截断...]'
    return summary + f'<pre class="ctx-pre">{body}</pre>'


def build_llm_answer(payload: dict) -> str:
    if payload.get("error"):
        return (
            f'<div class="llm-error">⚠️ LLM 调用失败</div>'
            f'<pre class="error-pre">错误：{html.escape(str(payload["error"]))}</pre>'
            f'<div class="fallback-box">兜底答案：{html.escape(payload.get("fallback", ""))}</div>'
        )
    parsed = payload.get("parsed") or {}
    answer = parsed.get("answer", "")
    confidence = parsed.get("confidence")
    reasoning = parsed.get("reasoning", "")
    fallback_flag = answer.startswith("基于记忆：")
    parts = []
    if fallback_flag:
        parts.append('<div class="llm-warn">⚠️ 检测到兜底字符串「基于记忆：」</div>')
    parts.append(f'<div class="answer-box"><b>最终答案</b>'
                 f'{f" <span class=\"conf\">置信度 {confidence}</span>" if confidence else ""}</div>')
    parts.append(f'<pre class="answer-pre">{html.escape(answer)}</pre>')
    if reasoning:
        parts.append(f'<details><summary>LLM 推理过程</summary>'
                     f'<pre class="reasoning-pre">{html.escape(reasoning)}</pre></details>')
    return "".join(parts)


def render_generic(payload) -> str:
    if isinstance(payload, (dict, list)):
        s = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        s = str(payload)
    if len(s) > 8000:
        s = s[:8000] + "\n...[截断]..."
    return f'<pre class="generic-pre">{html.escape(s)}</pre>'


def render_stage(name: str, payload) -> str:
    title = STAGE_CN.get(name, name)
    if name == "mas_scored":
        body = build_mas_table(payload or {})
    elif name == "final_context":
        body = build_final_context(payload or {})
    elif name == "llm_answer":
        body = build_llm_answer(payload or {})
    else:
        body = render_generic(payload)
    return f'<details class="stage" open="{name in ("mas_scored","final_context","llm_answer")}"><summary>{title} <code>{name}</code></summary><div class="stage-body">{body}</div></details>'


def build_pipeline_summary(trace: dict) -> str:
    """渲染管线时间线：召回 → 融合 → MAS → CBA → 上下文"""
    def cnt(stage):
        p = get_stage(trace, stage) or {}
        if stage == "fact_recall_fused":
            return len(p.get("fused_facts") or p.get("facts") or [])
        if stage == "fact_recall_per_path":
            paths = p.get("paths") or {}
            return sum(len(v) if isinstance(v, list) else 0 for v in paths.values())
        if stage == "graph_walk":
            return len(p.get("expanded_facts") or p.get("new_facts") or [])
        if stage == "mas_scored":
            return len((p or {}).get("facts") or p.get("scored_facts") or [])
        if stage == "cba_budget":
            return len(p.get("budgeted") or p.get("facts") or [])
        if stage == "final_context":
            return p.get("fact_count") or 0
        return None
    stages = [("草稿", "draft_recall"), ("各路径", "fact_recall_per_path"),
              ("融合", "fact_recall_fused"), ("图遍历", "graph_walk"),
              ("MAS", "mas_scored"), ("CBA", "cba_budget"), ("上下文", "final_context")]
    parts = []
    for label, sname in stages:
        n = cnt(sname)
        if n is None:
            parts.append(f'<span class="step">{label}<span class="n">?</span></span>')
        else:
            parts.append(f'<span class="step">{label}<span class="n">{n}</span></span>')
        if label != "上下文":
            parts.append('<span class="arrow">→</span>')
    return '<div class="pipeline">' + "".join(parts) + '</div>'


def build_question_card(qid: str, trace: dict, oe: dict, sc: dict,
                         p1a_sc: dict | None) -> str:
    q = oe.get("question", "")
    gt = oe.get("answer", "") or oe.get("gt_answer", "")
    stages = trace.get("stages") or {}
    llm = stages.get("llm_answer") or {}
    parsed = llm.get("parsed") or {}
    hyp = parsed.get("answer", "") if parsed else llm.get("fallback", "")
    is_fb = is_fallback(trace)

    # judge
    judge_lbl = sc.get("autoeval_label")
    judge_resp = sc.get("judge_response", "")
    p1a_lbl = (p1a_sc or {}).get("autoeval_label")

    anomalies = detect_anomalies(trace)
    anomaly_chips = "".join(f'<span class="chip">{a}</span>' for a in anomalies) or '<span class="chip ok">无异常</span>'

    # verdict badge
    if judge_lbl is True:
        verdict = '<span class="verdict pass">P1-B/C：通过 ✓</span>'
    else:
        verdict = '<span class="verdict fail">P1-B/C：失败 ✗</span>'
    if p1a_lbl is True:
        verdict += '<span class="verdict was-pass">P1-A：通过</span>'
    elif p1a_lbl is False:
        verdict += '<span class="verdict was-fail">P1-A：失败</span>'

    # top-3 MAS preview
    mas = stages.get("mas_scored") or {}
    mas_facts = mas.get("facts") or mas.get("scored_facts") or []
    top3_html = ""
    for i, f in enumerate(mas_facts[:3]):
        if not isinstance(f, dict):
            continue
        m = f.get("mas") or f.get("score") or 0
        c = f.get("content") or ""
        top3_html += f'<li><span class="mas-score">[{m:.3f}]</span> {html.escape(c[:180])}</li>'

    # stages rendering
    stage_html = []
    for name in STAGE_ORDER:
        if name in stages:
            stage_html.append(render_stage(name, stages[name]))
    stages_block = "".join(stage_html)

    return f'''
<div class="card" id="card-{qid}">
  <div class="card-header">
    <h2>{qid}</h2>
    {verdict}
  </div>
  <div class="anomaly-row">{anomaly_chips}</div>
  <div class="pipeline-box">{build_pipeline_summary(trace)}</div>
  <div class="qa">
    <div class="qa-item"><b>问题</b><div class="qa-text">{html.escape(q)}</div></div>
    <div class="qa-item"><b>GT 答案</b><div class="qa-text gt">{html.escape(gt)}</div></div>
    <div class="qa-item"><b>最终 HYP</b><div class="qa-text hyp{" fallback" if is_fb else ""}">{html.escape(hyp)}</div></div>
  </div>
  <div class="top-mas">
    <h3>Top-3 MAS facts（最高分命中）</h3>
    <ol>{top3_html}</ol>
  </div>
  <div class="judge">
    <h3>Judge 响应</h3>
    <pre class="judge-pre">{html.escape(judge_resp[-1500:])}</pre>
  </div>
  <div class="stages">
    <h3>17 阶段管线详情</h3>
    {stages_block}
  </div>
</div>
'''


def build_sidebar(items: list[dict]) -> str:
    rows = []
    for it in items:
        qid = it["qid"]
        preview = html.escape(it["question_preview"])
        verdict = it["verdict_class"]
        anomalies = it.get("anomalies", [])
        anomaly_short = " ".join(a.split()[0] for a in anomalies[:2]) if anomalies else ""
        rows.append(
            f'<a href="#card-{qid}" class="sidebar-item {verdict}">'
            f'<div class="sb-qid">{qid} <span class="sb-anom">{anomaly_short}</span></div>'
            f'<div class="sb-q">{preview}</div>'
            f'</a>'
        )
    return '<div class="sidebar">' + "".join(rows) + '</div>'


def build_html(items: list[dict], cards_html: str) -> str:
    total = len(items)
    passed = sum(1 for it in items if it["verdict_class"] == "pass")
    failed = total - passed
    fallback_count = sum(1 for it in items if it.get("is_fallback"))
    chips = []
    chips.append(f'<span class="summary-chip">总题数 <b>{total}</b></span>')
    chips.append(f'<span class="summary-chip pass">通过 <b>{passed}</b></span>')
    chips.append(f'<span class="summary-chip fail">失败 <b>{failed}</b></span>')
    if fallback_count:
        chips.append(f'<span class="summary-chip warn">LLM 兜底 <b>{fallback_count}</b></span>')

    css = """
* { box-sizing: border-box; }
body { margin:0; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; background:#f5f5f7; color:#1d1d1f; font-size:14px; }
.header { padding:16px 24px; background:#1d1d1f; color:#fff; position:sticky; top:0; z-index:100; }
.header h1 { margin:0; font-size:18px; font-weight:500; }
.header .summary { margin-top:8px; display:flex; gap:8px; flex-wrap:wrap; }
.summary-chip { padding:4px 12px; background:rgba(255,255,255,0.1); border-radius:14px; font-size:12px; }
.summary-chip.pass { background:rgba(48,209,88,0.25); }
.summary-chip.fail { background:rgba(255,99,71,0.25); }
.summary-chip.warn { background:rgba(255,159,10,0.25); }
.layout { display:grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 80px); }
.sidebar { background:#fff; border-right:1px solid #e5e5e7; overflow-y:auto; position:sticky; top:80px; height: calc(100vh - 80px); }
.sidebar-item { display:block; padding:12px 16px; border-bottom:1px solid #f0f0f2; text-decoration:none; color:#1d1d1f; }
.sidebar-item:hover { background:#f5f5f7; }
.sidebar-item.active { background:#e8f0fe; border-left:3px solid #1a73e8; }
.sb-qid { font-size:12px; font-family:monospace; color:#6e6e73; margin-bottom:2px; }
.sb-anom { font-size:11px; color:#ff9500; margin-left:6px; }
.sb-q { font-size:12px; color:#3a3a3c; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sidebar-item.pass .sb-qid { color:#30d158; }
.sidebar-item.fail .sb-qid { color:#ff6347; }
.main { padding:24px; overflow-y:auto; }
.card { background:#fff; border-radius:12px; padding:24px; margin-bottom:24px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }
.card-header { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.card-header h2 { margin:0; font-family:monospace; font-size:16px; }
.verdict { padding:4px 10px; border-radius:6px; font-size:12px; font-weight:500; }
.verdict.pass { background:#d4f4dd; color:#1a7a37; }
.verdict.fail { background:#ffe1d9; color:#a33b1a; }
.verdict.was-pass { background:#e8f0fe; color:#1a73e8; }
.verdict.was-fail { background:#f0f0f2; color:#6e6e73; }
.anomaly-row { margin-bottom:12px; display:flex; gap:6px; flex-wrap:wrap; }
.chip { padding:2px 8px; border-radius:10px; font-size:11px; background:#fff4e0; color:#a37300; border:1px solid #ffcf80; }
.chip.ok { background:#e8f7ed; color:#1a7a37; border-color:#b3e6c4; }
.pipeline-box { margin-bottom:16px; padding:10px 12px; background:#f5f5f7; border-radius:8px; }
.pipeline { display:flex; align-items:center; flex-wrap:wrap; gap:4px; }
.step { display:inline-flex; flex-direction:column; align-items:center; padding:4px 10px; background:#fff; border-radius:6px; border:1px solid #e5e5e7; min-width:50px; }
.step .n { font-weight:bold; color:#1a73e8; font-size:13px; }
.arrow { color:#6e6e73; }
.qa { margin-bottom:16px; }
.qa-item { margin-bottom:10px; }
.qa-item b { display:block; font-size:12px; color:#6e6e73; margin-bottom:4px; }
.qa-text { padding:10px 12px; background:#f5f5f7; border-radius:6px; font-size:13px; line-height:1.5; }
.qa-text.gt { background:#e8f7ed; border-left:3px solid #30d158; }
.qa-text.hyp { background:#fff4e0; border-left:3px solid #ff9500; }
.qa-text.hyp.fallback { background:#ffe1d9; border-left:3px solid #ff3b30; }
.top-mas, .judge { margin-bottom:16px; }
.top-mas h3, .judge h3, .stages h3 { font-size:13px; color:#6e6e73; margin:0 0 8px 0; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }
.top-mas ol { margin:0; padding-left:20px; }
.top-mas li { padding:4px 0; font-size:12px; line-height:1.5; }
.mas-score { font-family:monospace; color:#1a73e8; font-weight:bold; }
.judge-pre { background:#1d1d1f; color:#f5f5f7; padding:10px; border-radius:6px; font-size:11px; overflow-x:auto; max-height:300px; overflow-y:auto; }
.stages { border-top:1px solid #e5e5e7; padding-top:12px; }
details.stage { margin-bottom:8px; border:1px solid #e5e5e7; border-radius:6px; }
details.stage > summary { padding:8px 12px; cursor:pointer; font-weight:500; background:#f5f5f7; border-radius:6px; }
details.stage > summary code { color:#6e6e73; font-size:11px; margin-left:8px; }
details.stage[open] > summary { border-bottom-left-radius:0; border-bottom-right-radius:0; }
.stage-body { padding:10px 12px; }
.mas-table { width:100%; border-collapse:collapse; font-size:11px; }
.mas-table th { text-align:left; padding:4px 6px; background:#f5f5f7; border-bottom:1px solid #e5e5e7; }
.mas-table td { padding:4px 6px; border-bottom:1px solid #f0f0f2; vertical-align:top; }
.mas-table td.content { max-width:500px; }
.ctx-summary { display:flex; gap:16px; padding:6px 10px; background:#f5f5f7; border-radius:4px; margin-bottom:8px; font-size:12px; }
.ctx-pre, .generic-pre, .answer-pre, .reasoning-pre { background:#f5f5f7; padding:10px; border-radius:6px; font-size:11px; overflow-x:auto; max-height:400px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; }
.answer-box { font-size:12px; color:#6e6e73; margin-bottom:6px; }
.conf { color:#1a73e8; margin-left:8px; }
.answer-pre { background:#fff4e0; border-left:3px solid #ff9500; }
.llm-error, .llm-warn { padding:6px 10px; background:#ffe1d9; border-radius:4px; font-size:12px; color:#a33b1a; margin-bottom:6px; }
.error-pre { background:#ffe1d9; padding:6px; border-radius:4px; font-size:11px; }
.fallback-box { padding:6px 10px; background:#fff4e0; border-radius:4px; font-size:12px; }
.empty { color:#6e6e73; font-style:italic; }
"""
    payload_data = {"items": items}
    payload_json = json.dumps(payload_data, ensure_ascii=False)
    # 安全嵌入 JSON
    payload_json_escaped = payload_json.replace("</", "<\\/")

    sidebar_html = build_sidebar(items)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>5 题 P1-B/C 诊断面板</title>
<style>{css}</style>
</head>
<body>
<div class="header">
  <h1>P1-B/C 诊断面板 — 5 题剩余失败分析</h1>
  <div class="summary">{"".join(chips)}</div>
</div>
<div class="layout">
{sidebar_html}
<div class="main">
{cards_html}
</div>
</div>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--p1a-scored", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    oracle = load_oracle()
    p1bc_scored = load_scored(Path(args.scored))
    p1a_scored = load_scored(Path(args.p1a_scored)) if args.p1a_scored else {}

    items = []
    cards = []
    for jp in sorted(debug_dir.glob("debug_*.json")):
        with jp.open() as f:
            trace = json.load(f)
        qid = trace.get("qid", jp.stem.replace("debug_", ""))[:8]
        oe = oracle.get(qid, {})
        sc = p1bc_scored.get(qid, {})
        p1a_sc = p1a_scored.get(qid)

        verdict_class = "pass" if sc.get("autoeval_label") is True else "fail"
        anomalies = detect_anomalies(trace)
        is_fb = is_fallback(trace)

        items.append({
            "qid": qid,
            "question_preview": oe.get("question", "")[:60],
            "verdict_class": verdict_class,
            "anomalies": anomalies,
            "is_fallback": is_fb,
        })
        cards.append(build_question_card(qid, trace, oe, sc, p1a_sc))

    html_doc = build_html(items, "".join(cards))
    Path(args.out).write_text(html_doc, encoding="utf-8")
    print(f"已生成 {args.out}")
    print(f"题数: {len(items)} | 通过: {sum(1 for i in items if i['verdict_class']=='pass')}")


if __name__ == "__main__":
    main()

"""构建 LongMemEval 错题自包含 HTML Dashboard。

读取 15 个错题的 debug JSON trace + wrong list + scored JSONL,
emit 单个 HTML 文件,浏览器双击即可打开,无外部依赖。

用法:
  PYTHONPATH=. uv run python scripts/build_dashboard.py \
      --debug-dir /home/manjaro/tmp/debug_wrong \
      --wrong-list /home/manjaro/tmp/ssp_wrong_list.json \
      --scored /home/manjaro/tmp/ssp_debug_results.jsonl.scored.jsonl \
      --out /home/manjaro/tmp/dashboard.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

STAGE_ORDER = [
    "forgetting",
    "query_embed",
    "draft_recall",
    "keywords",
    "entity_recall",
    "fact_recall_per_path",
    "fact_recall_fused",
    "graph_walk",
    "contradiction",
    "mas_scored",
    "expected_entities",
    "iterative_recall",
    "reverse_entity_expansion",
    "sufficiency_check",
    "expand_keywords_via_graph",
    "cba_budget",
    "final_context",
    "llm_answer",
]

STAGE_LABELS = {
    "forgetting": "遗忘衰减",
    "query_embed": "查询嵌入",
    "draft_recall": "投机召回",
    "keywords": "关键词抽取",
    "entity_recall": "实体召回",
    "fact_recall_per_path": "分路召回",
    "fact_recall_fused": "融合召回",
    "graph_walk": "图游走扩展",
    "contradiction": "矛盾过滤",
    "mas_scored": "MAS 评分",
    "expected_entities": "预期实体",
    "iterative_recall": "迭代召回",
    "reverse_entity_expansion": "反向实体扩展",
    "sufficiency_check": "充分性检查",
    "expand_keywords_via_graph": "图扩展关键词",
    "cba_budget": "CBA 预算分配",
    "final_context": "最终上下文",
    "llm_answer": "LLM 答案生成",
}

FALLBACK_PREFIX = "基于记忆:"


def load_scored(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = rec.get("question_id")
        if qid:
            out[qid] = rec
    return out


def load_wrong_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_debug(debug_dir: Path, qid: str) -> dict | None:
    p = debug_dir / f"debug_{qid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"JSON 解码错误: {e}"}


def esc(s) -> str:
    """HTML escape any string, tolerant of None."""
    return html.escape(str(s) if s is not None else "", quote=True)


def get_stage(trace: dict | None, name: str) -> dict | None:
    if trace is None:
        return None
    stages = trace.get("stages", {})
    return stages.get(name)


def detect_anomalies(trace: dict | None) -> list[dict]:
    """扫描 trace,返回按严重度排序的异常列表。"""
    if trace is None:
        return [{"level": "red", "tag": "缺失", "detail": "未找到 Debug trace"}]

    anomalies: list[dict] = []

    # LLM 兜底 (red): answer 以 "基于记忆:" 开头 → main.py 兜底路径
    llm = get_stage(trace, "llm_answer") or {}
    parsed = llm.get("parsed") or {}
    answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    if isinstance(answer, str) and answer.startswith(FALLBACK_PREFIX):
        anomalies.append({
            "level": "red",
            "tag": "LLM 兜底",
            "detail": f"answer 走了 main.py:358 兜底路径 (LLM 调用失败): {answer[:80]}",
        })
    if llm.get("error"):
        anomalies.append({
            "level": "red",
            "tag": "LLM 错误",
            "detail": f"llm_answer.error: {llm['error'][:200]}",
        })

    # 上下文为空 (red)
    fc = get_stage(trace, "final_context") or {}
    if fc.get("fact_count", -1) == 0:
        anomalies.append({
            "level": "red",
            "tag": "上下文为空",
            "detail": "final_context.fact_count == 0",
        })

    # 迭代召回为空 (yellow)
    ir = get_stage(trace, "iterative_recall") or {}
    if ir.get("new_facts_count", -1) == 0:
        anomalies.append({
            "level": "yellow",
            "tag": "迭代召回为空",
            "detail": "iterative_recall.new_facts_count == 0,没有新事实召回",
        })

    # 缺失实体 (yellow)
    re_ = get_stage(trace, "reverse_entity_expansion") or {}
    missing = re_.get("missing_entities") or []
    if missing:
        anomalies.append({
            "level": "yellow",
            "tag": "缺失实体",
            "detail": f"预期实体在 top 召回中缺失: {', '.join(map(str, missing[:5]))}{' ...' if len(missing) > 5 else ''}",
        })

    # 充分性卡死 (yellow)
    sc = get_stage(trace, "sufficiency_check") or {}
    sc_parsed = sc.get("parsed") or {}
    ek = get_stage(trace, "expand_keywords_via_graph") or {}
    if (isinstance(sc_parsed, dict) and sc_parsed.get("sufficient") is False
            and ek.get("count", 0) == 0):
        anomalies.append({
            "level": "yellow",
            "tag": "充分性卡死",
            "detail": "sufficiency_check.sufficient=False 且 expand_keywords_via_graph.count==0",
        })

    # Top MAS 过低 (yellow)
    mas = get_stage(trace, "mas_scored") or {}
    facts = mas.get("facts") or []
    if facts and isinstance(facts[0], dict):
        top_mas = facts[0].get("mas", 0)
        if top_mas < 0.3:
            anomalies.append({
                "level": "yellow",
                "tag": "Top MAS 过低",
                "detail": f"mas_scored.facts[0].mas={top_mas:.3f} (<0.3)",
            })

    # 无召回结果 (red, severe)
    fused = get_stage(trace, "fact_recall_fused") or {}
    if fused.get("total", -1) == 0:
        anomalies.append({
            "level": "red",
            "tag": "零融合召回",
            "detail": "fact_recall_fused.total == 0,无任何召回结果",
        })

    level_order = {"red": 0, "yellow": 1}
    anomalies.sort(key=lambda a: level_order.get(a["level"], 99))
    return anomalies


def build_pipeline_timeline(trace: dict | None) -> list[dict]:
    """抽取关键节点的 fact 数量,用于时间线展示。

    节点是 cumulative total (不是 delta):
      投机    = 投机召回 facts_count
      融合    = 4 路 RRF 融合 total
      图扩展  = fused + graph_walk.expanded_count (graph walk 后的总数)
      MAS     = mas_scored.count (含 reverse expansion + iterative 召回结果)
      CBA     = cba_budget.fact_count (预算分配后)
      上下文  = final_context.fact_count (最终进入 LLM 上下文)
    """
    if trace is None:
        return []
    points: list[dict] = []

    draft = get_stage(trace, "draft_recall") or {}
    draft_n = draft.get("facts_count")
    if draft_n is not None:
        points.append({"stage": "draft_recall", "count": draft_n, "label": "投机"})

    fused = get_stage(trace, "fact_recall_fused") or {}
    fused_n = fused.get("total")
    if fused_n is not None:
        points.append({"stage": "fact_recall_fused", "count": fused_n, "label": "融合"})

    gw = get_stage(trace, "graph_walk") or {}
    expanded = gw.get("expanded_count", 0) or 0
    if fused_n is not None:
        after_graph = fused_n + expanded
        points.append({"stage": "graph_walk", "count": after_graph, "label": "图扩展"})

    mas = get_stage(trace, "mas_scored") or {}
    mas_n = mas.get("count")
    if mas_n is not None:
        points.append({"stage": "mas_scored", "count": mas_n, "label": "MAS"})

    cba = get_stage(trace, "cba_budget") or {}
    cba_n = cba.get("fact_count")
    if cba_n is not None:
        points.append({"stage": "cba_budget", "count": cba_n, "label": "CBA"})

    fc = get_stage(trace, "final_context") or {}
    fc_n = fc.get("fact_count")
    if fc_n is not None:
        points.append({"stage": "final_context", "count": fc_n, "label": "上下文"})

    # compute delta vs prev
    prev = None
    for p in points:
        p["delta"] = None if prev is None else p["count"] - prev
        prev = p["count"]
    return points


def render_mas_table(payload: dict) -> str:
    facts = payload.get("facts") or []
    weights = payload.get("weights") or {}
    if not facts:
        return '<p class="empty">无 MAS 评分事实</p>'

    rows = []
    for f in facts[:15]:
        if not isinstance(f, dict):
            continue
        content = esc((f.get("content") or "")[:120])
        full = esc(f.get("content") or "")
        mas = f.get("mas", 0)
        tier = f.get("tier", "-")
        rows.append(
            f'<tr>'
            f'<td class="num mas">{mas:.3f}</td>'
            f'<td class="num">{esc(tier)}</td>'
            f'<td class="num">{f.get("semantic_match", 0):.3f}</td>'
            f'<td class="num">{f.get("edge_weight", 0):.3f}</td>'
            f'<td class="num">{f.get("recency", 0):.3f}</td>'
            f'<td class="num">{f.get("tier_boost", 0):.3f}</td>'
            f'<td class="num">{f.get("activation", 0):.3f}</td>'
            f'<td class="content" title="{full}">{content}</td>'
            f'</tr>'
        )
    weights_str = ", ".join(f"{k}={v}" for k, v in weights.items()) if weights else ""
    return (
        f'<p class="meta">数量={payload.get("count", 0)}  权重: {esc(weights_str)}</p>'
        '<table class="mas-table">'
        '<thead><tr>'
        '<th>MAS</th><th>层级</th><th>语义</th><th>边权</th>'
        '<th>时近</th><th>层级↑</th><th>激活</th><th>内容</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_per_path(payload: dict) -> str:
    paths = ["bm25", "vec", "tag", "trigram"]
    lambdas = payload.get("lambdas") or {}
    cols = []
    for p in paths:
        items = payload.get(p) or []
        top5 = items[:5] if isinstance(items, list) else []
        cells = []
        for it in top5:
            if not isinstance(it, dict):
                continue
            content = esc((it.get("content") or "")[:80])
            score = it.get("score", 0)
            cells.append(
                f'<div class="path-item"><span class="num">{score:.3f}</span> '
                f'<span class="content">{content}</span></div>'
            )
        if not cells:
            cells = ['<div class="empty">—</div>']
        cols.append(
            f'<div class="path-col">'
            f'<h4>{p} <span class="lambda">λ={lambdas.get(p, "?")}</span></h4>'
            f'{"".join(cells)}'
            f'</div>'
        )
    return f'<div class="path-grid">{"".join(cols)}</div>'


def render_final_context(payload: dict) -> str:
    char_len = payload.get("char_len", 0)
    tokens = payload.get("approx_tokens", 0)
    fact_count = payload.get("fact_count", 0)
    context = payload.get("context") or ""
    return (
        f'<p class="meta">事实={fact_count}  字符={char_len}  ~tokens={tokens}</p>'
        f'<pre class="context-pre">{esc(context)}</pre>'
    )


def render_llm_answer(payload: dict) -> str:
    parsed = payload.get("parsed") or {}
    answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
    reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else ""
    error = payload.get("error")

    is_fallback = isinstance(answer, str) and answer.startswith(FALLBACK_PREFIX)
    fallback_cls = " fallback" if is_fallback else ""

    tabs = [
        ("解析结果", (
            f'<div class="parsed-answer{fallback_cls}">{esc(answer)}</div>'
            + (f'<p class="meta">置信度: {esc(confidence)}</p>' if confidence is not None else '')
            + (f'<p class="meta">推理过程: {esc(reasoning)}</p>' if reasoning else '')
            + (f'<p class="error">错误: {esc(error)}</p>' if error else '')
        )),
        ("系统提示", f'<pre>{esc(payload.get("system_prompt") or "")}</pre>'),
        ("用户消息", f'<pre>{esc(payload.get("user_message") or "")}</pre>'),
        ("原始响应", f'<pre>{esc(payload.get("raw_response") or "")}</pre>'),
    ]

    buttons = []
    bodies = []
    for i, (name, body) in enumerate(tabs):
        active = "active" if i == 0 else ""
        buttons.append(
            f'<button class="tab-btn {active}" '
            f'onclick="switchTab(this, \'llm-tab-{i}\')">{name}</button>'
        )
        bodies.append(f'<div class="tab-body {active}" id="llm-tab-{i}">{body}</div>')

    return (
        '<div class="tabs">'
        f'<div class="tab-buttons">{"".join(buttons)}</div>'
        f'{"".join(bodies)}'
        '</div>'
    )


def render_generic_json(payload) -> str:
    if payload is None:
        return '<p class="empty">无数据</p>'
    s = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return f'<pre class="json-pre">{esc(s)}</pre>'


CUSTOM_RENDERERS = {
    "mas_scored": render_mas_table,
    "fact_recall_per_path": render_per_path,
    "final_context": render_final_context,
    "llm_answer": render_llm_answer,
}


def render_stage_section(name: str, payload) -> str:
    label = STAGE_LABELS.get(name, name)
    if payload is None:
        return (
            f'<details><summary><span class="stage-name">{label}</span> '
            f'<code class="stage-id">{name}</code> '
            f'<span class="missing-tag">缺失</span></summary>'
            f'<p class="empty">此阶段无记录数据</p></details>'
        )
    renderer = CUSTOM_RENDERERS.get(name, render_generic_json)
    body = renderer(payload)
    return (
        f'<details>'
        f'<summary><span class="stage-name">{label}</span> '
        f'<code class="stage-id">{name}</code></summary>'
        f'<div class="stage-body">{body}</div>'
        f'</details>'
    )


def level_color(level: str) -> str:
    return {"red": "#ef4444", "yellow": "#f59e0b"}.get(level, "#9ca3af")


def build_sidebar(items: list[dict]) -> str:
    """items: [{qid, question, anomalies, missing}]"""
    lines = []
    for it in items:
        qid = esc(it["qid"])
        preview = esc(it["question"][:60] + ("…" if len(it["question"]) > 60 else ""))
        if it.get("missing"):
            lines.append(
                f'<a class="sidebar-item missing" onclick="selectQid(\'{qid}\')">'
                f'<div class="qid">{qid}</div>'
                f'<div class="preview">{preview}</div>'
                f'<span class="tag tag-red">缺失</span>'
                f'</a>'
            )
            continue
        anomalies = it.get("anomalies") or []
        top = anomalies[0] if anomalies else None
        if top:
            color = level_color(top["level"])
            tag_text = esc(top["tag"])
            tag_html = f'<span class="tag" style="background:{color}">{tag_text}</span>'
        else:
            tag_html = '<span class="tag tag-green">正常</span>'
        lines.append(
            f'<a class="sidebar-item" onclick="selectQid(\'{qid}\')">'
            f'<div class="qid">{qid}</div>'
            f'<div class="preview">{preview}</div>'
            f'{tag_html}'
            f'</a>'
        )
    return "".join(lines)


def build_header_chips(items: list[dict]) -> str:
    """聚合所有题的异常标签,header 展示总数。"""
    counts: dict[str, int] = {}
    level_by_tag: dict[str, str] = {}
    for it in items:
        if it.get("missing"):
            counts["缺失"] = counts.get("缺失", 0) + 1
            level_by_tag["缺失"] = "red"
            continue
        for a in it.get("anomalies") or []:
            key = a["tag"]
            counts[key] = counts.get(key, 0) + 1
            level_by_tag[key] = a["level"]

    if not counts:
        return '<div class="chips"><span class="chip chip-green">未检测到异常</span></div>'

    chips = []
    for tag, n in sorted(counts.items(), key=lambda x: -x[1]):
        level = level_by_tag.get(tag, "yellow")
        color = level_color(level)
        chips.append(
            f'<span class="chip" style="background:{color}">{n}× {esc(tag)}</span>'
        )
    return f'<div class="chips">{"".join(chips)}</div>'


def build_question_html(qid: str, wrong: dict, scored: dict | None,
                        trace: dict | None) -> str:
    """渲染某题的完整主面板 HTML (server-side)。"""
    anomalies = detect_anomalies(trace)
    timeline = build_pipeline_timeline(trace)

    # Top card
    question = esc(wrong.get("question") or "")
    answer_gt = esc(wrong.get("answer_gt") or "")
    hypothesis = esc(wrong.get("hypothesis") or "")

    if scored:
        judge_resp = esc(scored.get("judge_response") or "")
        is_yes = scored.get("autoeval_label", False)
        badge_cls = "badge-no" if not is_yes else "badge-yes"
        badge_text = "错误 (NO)" if not is_yes else "正确 (YES)"
        judge_html = (
            f'<div class="judge-badge {badge_cls}">{badge_text}</div>'
            f'<div class="judge-raw"><span class="meta">Judge 响应:</span> {judge_resp}</div>'
        )
    else:
        judge_html = '<div class="judge-missing">无评分记录</div>'

    # Anomalies
    if anomalies:
        anom_chips = []
        for a in anomalies:
            color = level_color(a["level"])
            anom_chips.append(
                f'<div class="anomaly" style="border-left-color:{color}">'
                f'<span class="anomaly-tag" style="background:{color}">{esc(a["tag"])}</span>'
                f'<span class="anomaly-detail">{esc(a["detail"])}</span>'
                f'</div>'
            )
        anomalies_html = (
            '<div class="anomaly-section">'
            f'<h3>⚠️ 异常 ({len(anomalies)})</h3>'
            f'{"".join(anom_chips)}'
            '</div>'
        )
    else:
        anomalies_html = '<div class="anomaly-section"><h3>✅ 无异常</h3></div>'

    # Timeline
    if timeline:
        tl_nodes = []
        for p in timeline:
            delta = p["delta"]
            if delta is None:
                delta_html = ""
            elif delta > 0:
                delta_html = f'<span class="delta delta-up">+{delta}</span>'
            elif delta < 0:
                delta_html = f'<span class="delta delta-down">{delta}</span>'
            else:
                delta_html = '<span class="delta">±0</span>'
            tl_nodes.append(
                f'<div class="tl-node"><div class="tl-label">{esc(p["label"])}</div>'
                f'<div class="tl-count">{p["count"]}</div>{delta_html}</div>'
            )
        interleaved = []
        for i, n in enumerate(tl_nodes):
            interleaved.append(n)
            if i < len(tl_nodes) - 1:
                interleaved.append('<div class="tl-arrow">→</div>')
        timeline_html = (
            '<div class="timeline">'
            f'{"".join(interleaved)}'
            '</div>'
        )
    else:
        timeline_html = '<div class="timeline empty">(无 trace)</div>'

    # Stages accordion
    stages = (trace or {}).get("stages", {})
    sections = []
    for name in STAGE_ORDER:
        sections.append(render_stage_section(name, stages.get(name)))
    stages_html = f'<div class="stages">{"".join(sections)}</div>'

    return (
        f'<div class="qcard">'
        f'<div class="qcard-row"><span class="meta-label">题目 ID:</span> <code>{qid}</code></div>'
        f'<div class="qcard-row"><span class="meta-label">问题:</span><div class="long-text">{question}</div></div>'
        f'<div class="qcard-row"><span class="meta-label">标准答案:</span><div class="long-text gt-text">{answer_gt}</div></div>'
        f'<div class="qcard-row"><span class="meta-label">模型答案:</span><div class="long-text hyp-text">{hypothesis}</div></div>'
        f'<div class="qcard-row judge-row">{judge_html}</div>'
        f'</div>'
        f'{anomalies_html}'
        f'<div class="section"><h3>管线时间线</h3>{timeline_html}</div>'
        f'<div class="section"><h3>阶段 Trace ({len(STAGE_ORDER)})</h3>{stages_html}</div>'
    )


def build_html(items: list[dict], by_qid: dict[str, dict]) -> str:
    """items: sidebar items (qid, question, anomalies, missing)
    by_qid: {qid: {meta, anomalies, timeline, html}}
    """
    header_chips = build_header_chips(items)
    sidebar = build_sidebar(items)

    payload = {
        "qids": [it["qid"] for it in items],
        "by_qid": by_qid,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    # Prevent </script> in payload from breaking the <script type="application/json">
    # block. JSON allows escaped forward slash, so </  becomes <\/  which is safe.
    payload_json = payload_json.replace("</", "<\\/")

    total = len(items)
    missing_count = sum(1 for it in items if it.get("missing"))

    summary_parts = [f"共 {total} 题"]
    if missing_count:
        summary_parts.append(f"· 缺失 trace: {missing_count}")
    summary_parts.append("· 点击左侧题目查看其 17 阶段管线全流程 trace")
    summary_html = " ".join(summary_parts)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>AegisMem 错题 Dashboard — LongMemEval 单会话偏好类</title>
<style>
:root {{
  --bg: #0f1115;
  --bg-elev: #181b22;
  --bg-elev2: #1f232c;
  --border: #2a2f3a;
  --text: #e4e7eb;
  --text-dim: #9ca3af;
  --red: #ef4444;
  --yellow: #f59e0b;
  --green: #10b981;
  --blue: #3b82f6;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--text); font-size: 14px;
}}
header {{
  background: var(--bg-elev); border-bottom: 1px solid var(--border);
  padding: 16px 24px; position: sticky; top: 0; z-index: 10;
}}
header h1 {{ margin: 0 0 8px 0; font-size: 18px; }}
header .summary {{ color: var(--text-dim); margin-bottom: 8px; font-size: 13px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{
  padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
  color: #fff; white-space: nowrap;
}}
.chip-green {{ background: var(--green); }}
.layout {{
  display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 100px);
}}
.sidebar {{
  background: var(--bg-elev); border-right: 1px solid var(--border);
  overflow-y: auto; max-height: calc(100vh - 100px); position: sticky; top: 100px;
}}
.sidebar-item {{
  display: block; padding: 10px 14px; border-bottom: 1px solid var(--border);
  cursor: pointer; text-decoration: none; color: var(--text);
  transition: background 0.1s;
}}
.sidebar-item:hover {{ background: var(--bg-elev2); }}
.sidebar-item.active {{ background: var(--bg-elev2); border-left: 3px solid var(--blue); }}
.sidebar-item.missing {{ opacity: 0.6; }}
.sidebar-item .qid {{ font-family: monospace; font-size: 12px; color: var(--text-dim); }}
.sidebar-item .preview {{ font-size: 12px; margin-top: 3px; line-height: 1.3; }}
.sidebar-item .tag {{
  display: inline-block; margin-top: 5px; padding: 2px 8px; border-radius: 10px;
  font-size: 10px; font-weight: 600; color: #fff;
}}
.tag-red {{ background: var(--red); }}
.tag-yellow {{ background: var(--yellow); color: #000; }}
.tag-green {{ background: var(--green); }}
.main {{ padding: 24px; overflow-x: auto; }}
.qcard {{
  background: var(--bg-elev); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
}}
.qcard-row {{ margin-bottom: 12px; }}
.qcard-row:last-child {{ margin-bottom: 0; }}
.meta-label {{ color: var(--text-dim); font-size: 12px; display: inline-block; min-width: 90px; vertical-align: top; }}
.long-text {{ display: inline-block; max-width: calc(100% - 100px); white-space: pre-wrap; word-break: break-word; }}
.gt-text {{ color: var(--green); }}
.hyp-text {{ color: var(--text); }}
.judge-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.judge-badge {{ padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; }}
.badge-no {{ background: var(--red); color: #fff; }}
.badge-yes {{ background: var(--green); color: #fff; }}
.judge-raw {{ font-size: 12px; color: var(--text-dim); }}
.judge-raw .meta {{ color: var(--text-dim); }}
.section {{
  background: var(--bg-elev); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
}}
.section h3 {{ margin: 0 0 12px 0; font-size: 14px; color: var(--text-dim); }}
.anomaly-section h3 {{ color: var(--yellow); }}
.anomaly {{
  background: var(--bg-elev2); border-left: 4px solid var(--yellow);
  padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;
  display: flex; gap: 10px; align-items: flex-start;
}}
.anomaly-tag {{
  padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
  color: #fff; flex-shrink: 0; min-width: 110px; text-align: center;
}}
.anomaly-detail {{ font-size: 12px; line-height: 1.4; word-break: break-word; }}
.timeline {{
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  padding: 12px; background: var(--bg-elev2); border-radius: 6px;
}}
.timeline.empty {{ color: var(--text-dim); padding: 16px; }}
.tl-node {{
  text-align: center; padding: 6px 12px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px; min-width: 70px;
}}
.tl-label {{ font-size: 11px; color: var(--text-dim); }}
.tl-count {{ font-size: 18px; font-weight: 700; }}
.tl-arrow {{ color: var(--text-dim); }}
.delta {{ font-size: 10px; margin-top: 2px; display: block; }}
.delta-up {{ color: var(--green); }}
.delta-down {{ color: var(--red); }}
.stages details {{
  border: 1px solid var(--border); border-radius: 4px; margin-bottom: 4px;
  background: var(--bg-elev2);
}}
.stages summary {{
  cursor: pointer; padding: 8px 12px; font-family: monospace; font-size: 13px;
}}
.stages summary:hover {{ background: var(--bg); }}
.stage-name {{ color: var(--blue); font-family: -apple-system, "PingFang SC", sans-serif; }}
.stage-id {{ color: var(--text-dim); font-size: 11px; margin-left: 8px; }}
.missing-tag {{ color: var(--red); margin-left: 8px; font-size: 11px; }}
.stage-body {{ padding: 12px; border-top: 1px solid var(--border); }}
.json-pre, .context-pre {{
  margin: 0; white-space: pre-wrap; word-break: break-word;
  font-family: "JetBrains Mono", "Consolas", monospace; font-size: 11px;
  max-height: 500px; overflow-y: auto; background: var(--bg);
  padding: 10px; border-radius: 4px; color: var(--text);
}}
.context-pre {{ color: var(--text-dim); }}
.meta {{ font-size: 12px; color: var(--text-dim); margin: 4px 0; }}
.empty {{ color: var(--text-dim); font-style: italic; padding: 8px; }}
.error {{ color: var(--red); font-size: 12px; }}
.mas-table {{
  width: 100%; border-collapse: collapse; font-size: 12px;
}}
.mas-table th, .mas-table td {{
  border: 1px solid var(--border); padding: 4px 8px; text-align: left;
}}
.mas-table th {{ background: var(--bg-elev); color: var(--text-dim); font-weight: 600; }}
.mas-table td.num {{ font-family: monospace; text-align: right; }}
.mas-table td.mas {{ font-weight: 700; color: var(--blue); }}
.mas-table td.content {{ max-width: 400px; }}
.path-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.path-col {{ background: var(--bg); padding: 10px; border-radius: 4px; }}
.path-col h4 {{ margin: 0 0 8px 0; font-size: 12px; color: var(--blue); }}
.lambda {{ color: var(--text-dim); font-size: 11px; font-weight: 400; }}
.path-item {{ font-size: 11px; margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px dashed var(--border); }}
.path-item .num {{ font-family: monospace; color: var(--green); margin-right: 6px; }}
.path-item .content {{ color: var(--text); }}
.tabs {{ }}
.tab-buttons {{ display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }}
.tab-btn {{
  background: var(--bg-elev2); border: 1px solid var(--border); border-bottom: none;
  color: var(--text-dim); padding: 6px 14px; cursor: pointer; font-size: 12px;
  border-radius: 4px 4px 0 0;
}}
.tab-btn.active {{ background: var(--bg); color: var(--text); border-bottom: 1px solid var(--bg); }}
.tab-body {{ display: none; }}
.tab-body.active {{ display: block; }}
.parsed-answer {{
  background: var(--bg); padding: 10px; border-radius: 4px;
  white-space: pre-wrap; word-break: break-word;
}}
.parsed-answer.fallback {{ border-left: 4px solid var(--red); color: var(--red); }}
.tab-body pre {{
  margin: 0; white-space: pre-wrap; word-break: break-word;
  font-family: "JetBrains Mono", "Consolas", monospace; font-size: 11px;
  max-height: 500px; overflow-y: auto; background: var(--bg);
  padding: 10px; border-radius: 4px;
}}
.placeholder {{ padding: 40px; text-align: center; color: var(--text-dim); }}
</style>
</head>
<body>
<header>
  <h1>AegisMem 错题 Dashboard — LongMemEval 单会话偏好类</h1>
  <div class="summary">{summary_html}</div>
  {header_chips}
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar">{sidebar}</nav>
  <main class="main" id="main">
    <div class="placeholder">← 从左侧选择题目查看其完整管线 trace</div>
  </main>
</div>
<script type="application/json" id="payload">{payload_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);

function selectQid(qid) {{
  // toggle active state
  document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
  const items = document.querySelectorAll('.sidebar-item');
  items.forEach(el => {{
    const qidDiv = el.querySelector('.qid');
    if (qidDiv && qidDiv.textContent === qid) el.classList.add('active');
  }});
  const entry = DATA.by_qid[qid];
  const main = document.getElementById('main');
  if (!entry) {{
    main.innerHTML = '<div class="placeholder">无 qid=' + qid + ' 的数据</div>';
    return;
  }}
  main.innerHTML = entry.html;
  window.scrollTo(0, 0);
}}

function switchTab(btn, bodyId) {{
  const tabs = btn.closest('.tabs');
  tabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  tabs.querySelectorAll('.tab-body').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(bodyId).classList.add('active');
}}

// auto-select first question on load
document.addEventListener('DOMContentLoaded', () => {{
  if (DATA.qids.length > 0) selectQid(DATA.qids[0]);
}});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="构建自包含错题 Dashboard HTML")
    parser.add_argument("--debug-dir", type=Path,
                        default=Path("/home/manjaro/tmp/debug_wrong"))
    parser.add_argument("--wrong-list", type=Path,
                        default=Path("/home/manjaro/tmp/ssp_wrong_list.json"))
    parser.add_argument("--scored", type=Path,
                        default=Path("/home/manjaro/tmp/ssp_debug_results.jsonl.scored.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/manjaro/tmp/dashboard.html"))
    args = parser.parse_args()

    if not args.wrong_list.exists():
        print(f"错误:错题列表未找到: {args.wrong_list}", file=sys.stderr)
        sys.exit(2)

    wrong = load_wrong_list(args.wrong_list)
    scored = load_scored(args.scored)
    print(f"已加载 {len(wrong)} 条错题,{len(scored)} 条评分记录",
          file=sys.stderr)

    items: list[dict] = []
    by_qid: dict[str, dict] = {}

    for entry in wrong:
        qid = entry["qid"]
        trace = load_debug(args.debug_dir, qid)
        missing = trace is None
        anomalies = detect_anomalies(trace)
        timeline = build_pipeline_timeline(trace)
        html_body = build_question_html(qid, entry, scored.get(qid), trace)
        items.append({
            "qid": qid,
            "question": entry.get("question") or "",
            "anomalies": anomalies,
            "missing": missing,
        })
        by_qid[qid] = {
            "html": html_body,
            "anomalies": anomalies,
            "timeline": timeline,
        }

    html_doc = build_html(items, by_qid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_doc, encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"已写入 {args.out} ({size_kb:.1f} KB)", file=sys.stderr)

    # Summary
    n_missing = sum(1 for it in items if it["missing"])
    n_with_anomaly = sum(1 for it in items if not it["missing"] and it["anomalies"])
    print(f"汇总:共 {len(items)} 题,{n_missing} 条缺失 trace,"
          f"{n_with_anomaly} 条有异常", file=sys.stderr)

    # Aggregate anomaly counts
    counts: dict[str, int] = {}
    for it in items:
        for a in it.get("anomalies") or []:
            counts[a["tag"]] = counts.get(a["tag"], 0) + 1
    if counts:
        print("异常分布:", file=sys.stderr)
        for tag, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {tag:25s} {n}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
上下文预算分配模块 — 根据 MAS 分数分配 Token 预算并构造最终上下文。

职责：
  1. 委托 ContextBudgetAllocator 进行比例分配
  2. 将分配结果格式化为 LLM 可读的上下文字符串
  3. 可选拼接最近对话轮次
"""

from __future__ import annotations

import logging

from internal.domain.model.buffer import DialogTurn, MessageBuffer
from internal.domain.model.fact import Fact
from internal.domain.services.context_budget_allocator import ContextBudgetAllocator
from internal.util.debug_collector import DebugCollector

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """粗略估算中英文混合文本的 token 数（1 token ~ 2 chars）。"""
    return max(1, len(text) // 2)


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    """按 token 预算截断文本。"""
    if _estimate_tokens(text) <= max_tokens:
        return text
    # 粗略截断字符
    return text[: max_tokens * 2]


class CBAService:
    """上下文预算分配服务层。"""

    def __init__(self) -> None:
        self._allocator = ContextBudgetAllocator()

    async def allocate(
        self,
        scored_facts: list[tuple[Fact, float]],
        total_budget: int | None = None,
        *,
        debug: DebugCollector | None = None,
    ) -> list[tuple[Fact, int]]:
        if total_budget is None:
            total_budget = 4000
        allocated = self._allocator.allocate(scored_facts, total_budget)
        if debug is not None:
            debug.record("cba_budget", {
                "total_budget": total_budget,
                "fact_count": len(allocated),
                "allocations": [
                    {
                        "id": str(f.id),
                        "content": (f.content or "")[:80],
                        "tokens": int(b),
                        "happened_at": f.metadata.HappendTime.isoformat() if (f.metadata and f.metadata.HappendTime) else None,
                    }
                    for f, b in allocated
                ],
            })
        return allocated

    async def build_retrieval_context(
        self,
        scored_facts_with_budget: list[tuple[Fact, int]],
        dialog_buffer: MessageBuffer | None = None,
        *,
        debug: DebugCollector | None = None,
    ) -> str:
        """构建最终记忆上下文字符串。

        每条事实附带：created_at、metadata（Person/Object/Location/Event/
        Organization/Preference/HappendTime/MentionedTime/History）、original_msg、tags。
        事实按 created_at 升序排列，便于 LLM 做时序推理。
        """
        parts = ["[记忆片段]"]

        # 按 created_at 升序排，便于时序推理；缺失时间的放在最后
        sorted_facts = sorted(
            scored_facts_with_budget,
            key=lambda fb: (fb[0].created_at is None, fb[0].created_at),
        )

        for fact, budget in sorted_facts:
            content_budget = max(budget // 2, 80)
            orig_budget = max(budget - content_budget, 40)

            ts = fact.created_at.isoformat() if fact.created_at else "unknown"
            truncated = _truncate_by_tokens(fact.content, content_budget)

            tag_names = [t.Entity.name for t in fact.tag]
            tags_str = f"\n  tags: {', '.join(tag_names)}" if tag_names else ""

            meta_lines = _format_metadata(fact.metadata)
            meta_str = f"\n  metadata:\n{meta_lines}" if meta_lines else ""

            orig_str = ""
            if fact.original_msg:
                orig_truncated = _truncate_by_tokens(fact.original_msg, orig_budget)
                orig_str = f"\n  original_msg: {orig_truncated}"

            parts.append(
                f"- [{ts}] fact: {truncated}{tags_str}{meta_str}{orig_str}"
            )

        if dialog_buffer and dialog_buffer.size > 0:
            parts.append("\n[最近对话]")
            for turn in dialog_buffer.get_recent(10):
                parts.append(f"{turn.role}: {turn.content}")

        result = "\n".join(parts)
        if debug is not None:
            debug.record("final_context", {
                "char_len": len(result),
                "approx_tokens": _estimate_tokens(result),
                "fact_count": len(scored_facts_with_budget),
                "context": result,
            })
        return result


def _format_metadata(meta) -> str:
    """把 Metadata 中非空字段格式化为多行缩进文本。"""
    if meta is None:
        return ""
    fields = [
        ("Person", meta.Person),
        ("Object", meta.Object),
        ("Location", meta.Location),
        ("Event", meta.Event),
        ("Organization", meta.Organization,),
        ("Preference", meta.Preference),
    ]
    lines: list[str] = []
    for name, val in fields:
        if val:
            lines.append(f"    {name}: {val}")
    if meta.HappendTime:
        lines.append(f"    HappendTime: {meta.HappendTime.isoformat()}")
    if meta.MentionedTime:
        lines.append(f"    MentionedTime: {meta.MentionedTime.isoformat()}")
    if meta.History:
        history_str = "; ".join(meta.History)
        lines.append(f"    History: {history_str}")
    return "\n".join(lines)
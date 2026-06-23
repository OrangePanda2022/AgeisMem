"""单题调用追踪：以 contextvar 作用域记录一道题在 retrieve 管线中各工具的调用情况。

设计：
  - CallTrace 是一组可变计数器，由 answer() 在入口创建并 set 进 ContextVar。
  - 各工具（LLM/Embedding/Recall/GraphWalk/MAS/CBA）通过 get_trace() 取当前题的 trace，
    命中即自增对应计数器；非追踪上下文（如 ingest、CLI ad-hoc 调用）get_trace() 返回
    None，所有钩子都 `if (t := get_trace()) is not None:` 守护，零开销。
  - asyncio 每个任务拷贝一份 context（copy-on-write），故 eval 的并发题目互不污染：
    contextvar 仅存对象引用，CallTrace 在各任务的钩子里原地自增，指向各自的对象。

输出：to_dict() 产出 4 个分桶（reasoning_chain / recall_calls / graph_walk /
tool_calls）。每个数据只出现一次，无跨桶重复（详见 plan 的 De-overlap note）。

注意：embedding.cache_hits 反映进程级缓存状态（embedding_client 是跨题共享的单例），
非每题全新冷启动的命中数。
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class CallTrace:
    """单题 retrieve 管线调用计数器。所有字段在钩子里原地自增。"""

    qid: str = ""

    # ---- reasoning_chain（推理链）----
    stages: list[str] = field(default_factory=list)  # 有序执行轨迹（字面"推理链"）
    iterative_rounds: int = 0  # 充分性检查推理循环实际执行的轮数（0~max_rounds）
    # LLM 内部 chain-of-thought：保留每次 generate 调用的 reasoning 原文（CoT）
    llm_reasoning_texts: list[str] = field(default_factory=list)

    # ---- recall_calls ----
    recall_calls: int = 0  # _keyword_recall（4 路）触发次数
    draft_calls: int = 0  # _draft_recall（投机草稿）触发次数

    # ---- graph_walk ----
    graph_walk_calls: int = 0
    graph_walk_nodes_visited: int = 0  # 含种子节点
    graph_walk_nodes_expanded: int = 0  # 仅新增节点

    # ---- recall_facts（每路找回 fact 数，跨多次召回累计，与 recall_calls 对应）----
    # _keyword_recall 四路 + _draft_recall 的 vec 一路；fused 为 RRF 融合去重后数。
    recall_facts_bm25: int = 0
    recall_facts_trigram: int = 0
    recall_facts_vec: int = 0
    recall_facts_tag: int = 0
    recall_facts_fused: int = 0

    # ---- final_context（最终喂给 LLM 的上下文，整题唯一非累计）----
    final_fact_count: int = 0  # CBA 预算分配后进入 context 的 fact 总数
    final_context_char_length: int = 0  # 最终 context 字符长度

    # ---- tool_calls ----
    llm_total: int = 0  # generate() API 次数（含 debate 扇出的多次 generate）
    llm_by_method: dict[str, int] = field(default_factory=dict)  # 高层方法语义计数
    embedding_api_calls: int = 0
    embedding_cache_hits: int = 0
    embedding_texts: int = 0
    mas_calls: int = 0
    cba_allocate: int = 0
    cba_build_context: int = 0

    def mark(self, name: str) -> None:
        """在 stages 轨迹里追加一个执行步骤（与计数器自增配套使用）。"""
        self.stages.append(name)

    def llm_method(self, name: str) -> None:
        """高层 LLM 方法命中：by_method[name] += 1 并 mark。"""
        self.llm_by_method[name] = self.llm_by_method.get(name, 0) + 1
        self.stages.append(name)

    def add_recall_facts(self, *, bm25: int = 0, trigram: int = 0,
                         vec: int = 0, tag: int = 0, fused: int = 0) -> None:
        """累加一次召回里各路找回的 fact 数 + RRF 融合去重后数。

        _keyword_recall 传满四路+fused；_draft_recall 仅传 vec+fused（投机一路）。
        一题多次召回（主/反向扩展/迭代轮）累计，与 recall_calls 计数对应。
        """
        self.recall_facts_bm25 += bm25
        self.recall_facts_trigram += trigram
        self.recall_facts_vec += vec
        self.recall_facts_tag += tag
        self.recall_facts_fused += fused

    def to_dict(self) -> dict:
        """产出 4 分桶结构；调用方（eval 脚本）负责 prepend question_id。"""
        return {
            "qid": self.qid,
            "reasoning_chain": {
                "iterative_rounds": self.iterative_rounds,
                "stages": list(self.stages),
                "llm_reasoning": {
                    "calls": len(self.llm_reasoning_texts),
                    "total_chars": sum(len(s) for s in self.llm_reasoning_texts),
                    "texts": list(self.llm_reasoning_texts),
                },
            },
            "recall_calls": {
                "keyword_4path": self.recall_calls,
                "draft": self.draft_calls,
                "total": self.recall_calls + self.draft_calls,
            },
            "recall_facts": {
                "per_method": {
                    "bm25": self.recall_facts_bm25,
                    "trigram": self.recall_facts_trigram,
                    "vec": self.recall_facts_vec,
                    "tag": self.recall_facts_tag,
                },
                "fused": self.recall_facts_fused,
            },
            "graph_walk": {
                "calls": self.graph_walk_calls,
                "nodes_visited": self.graph_walk_nodes_visited,
                "nodes_expanded": self.graph_walk_nodes_expanded,
            },
            "final_context": {
                "fact_count": self.final_fact_count,
                "char_length": self.final_context_char_length,
            },
            "tool_calls": {
                "llm": {
                    "total": self.llm_total,
                    "by_method": dict(self.llm_by_method),
                },
                "embedding": {
                    "api_calls": self.embedding_api_calls,
                    "cache_hits": self.embedding_cache_hits,
                    "texts": self.embedding_texts,
                },
                "mas": self.mas_calls,
                "cba": {
                    "allocate": self.cba_allocate,
                    "build_context": self.cba_build_context,
                },
            },
        }


_current: ContextVar[CallTrace | None] = ContextVar("aegismem_call_trace", default=None)


def get_trace() -> CallTrace | None:
    """取当前题的 CallTrace；非追踪上下文（ingest/CLI）返回 None。"""
    return _current.get()


@contextmanager
def trace_scope(trace: CallTrace) -> Iterator[CallTrace]:
    """设置当前题的 trace 作用域；退出（return/raise）时自动 reset，防 contextvar 泄漏。

    用法（在 answer() 入口）：
        trace = CallTrace(qid=qid)
        self.last_trace = trace
        with trace_scope(trace):
            return await self._answer_body(...)
    """
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)

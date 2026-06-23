"""单题调试收集器：把召回管线每一步的输入/输出捞到一个 JSON 文件。

用法：
    debug = DebugCollector(qid="0bb5a684", query="...")
    debug.record("keywords", {"raw": [...], "fallback": False})
    ...
    debug.dump("~/tmp/debug_xxx.json")

设计原则：
    - 不引入全局状态：实例由 pipeline 入口构造，显式传给各 service。
    - debug=None 时，所有 record 调用应被业务侧 if-guard 跳过，零开销。
    - 默认 JSON 化时对非基础类型（datetime/UUID/np 标量）尽力序列化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID


def _default(o: Any) -> Any:
    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, UUID):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


@dataclass
class DebugCollector:
    qid: str
    query: str
    stages: dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, payload: dict) -> None:
        """记录一个 stage 的 payload。同名 stage 会被覆盖。"""
        self.stages[stage] = payload

    def dump(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "qid": self.qid,
            "query": self.query,
            "stages": self.stages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=_default)

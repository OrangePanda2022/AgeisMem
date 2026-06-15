"""全局 token 用量计数器 + 文件日志。

进程内三客户端（llm/embedding/judge）每次调用后调用 tracker.add(...)：
  - 累加到内存计数器（用于 run 末汇总）
  - 追加一行 JSONL 到 /home/manjaro/AI/TokenLog/YYYY-MM-DD/{model}_HHMMSS.jsonl

文件粒度：按进程启动日期分文件夹，文件名后缀是进程启动时间（HHMMSS）。
同一进程的 3 个模型文件同组，跨进程互不覆盖。
runtime summary：调用 tracker.write_run_summary(run_label) 在每个文件末尾再
追加一行 type=summary 的快照。

线程安全：threading.Lock 保护并发写。
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LOG_ROOT = Path("/home/manjaro/AI/TokenLog")


@dataclass
class _ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


@dataclass
class _Tracker:
    llm: _ModelUsage = field(default_factory=_ModelUsage)
    embedding: _ModelUsage = field(default_factory=_ModelUsage)
    judge: _ModelUsage = field(default_factory=_ModelUsage)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pid: int = field(default_factory=os.getpid)
    _start: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        day = self._start.strftime("%Y-%m-%d")
        self._log_dir = LOG_ROOT / day
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._suffix = self._start.strftime("%H%M%S")

    def _log_path(self, model: str) -> Path:
        return self._log_dir / f"{model}_{self._suffix}.jsonl"

    def add(self, model: str, *, prompt: int = 0, completion: int = 0, total: int = 0) -> None:
        slot = getattr(self, model, None)
        if slot is None:
            return
        total = total or (prompt + completion)
        with self._lock:
            slot.prompt_tokens += prompt
            slot.completion_tokens += completion
            slot.total_tokens += total
            slot.calls += 1
            try:
                with self._log_path(model).open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "pid": self._pid,
                        "type": "call",
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": total,
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def write_run_summary(self, run_label: str = "") -> None:
        with self._lock:
            for model, slot in (("llm", self.llm), ("embedding", self.embedding), ("judge", self.judge)):
                try:
                    with self._log_path(model).open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "pid": self._pid,
                            "type": "summary",
                            "run_label": run_label,
                            "calls": slot.calls,
                            "prompt_tokens": slot.prompt_tokens,
                            "completion_tokens": slot.completion_tokens,
                            "total_tokens": slot.total_tokens,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass

    def reset(self) -> None:
        with self._lock:
            self.llm = _ModelUsage()
            self.embedding = _ModelUsage()
            self.judge = _ModelUsage()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "llm": vars(self.llm).copy(),
                "embedding": vars(self.embedding).copy(),
                "judge": vars(self.judge).copy(),
            }


tracker = _Tracker()

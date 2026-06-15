"""
预算分配器：根据 Fact 的 MAS 分数进行 Token 预算分配。
"""

from __future__ import annotations

from internal.domain.model.fact import Fact


class ContextBudgetAllocator:
    """根据 MAS 分数按比例分配 token 预算给各 Fact。"""

    MIN_BUDGET_PER_FACT = 100
    MAX_BUDGET_PER_FACT = 2000

    def allocate(
        self,
        scored_facts: list[tuple[Fact, float]],
        total_token_budget: int,
    ) -> list[tuple[Fact, int]]:
        """按 MAS 分数比例分配 token 预算。

        参数:
            scored_facts: (Fact, mas_score) 列表，按 MAS 降序。
            total_token_budget: 总 token 预算。

        返回:
            每个 Fact 及其分配的 token 数，按 MAS 降序。
        """
        if not scored_facts:
            return []

        total_mas = sum(s for _, s in scored_facts)
        if total_mas <= 0:
            eq = total_token_budget // len(scored_facts)
            return [(f, max(eq, self.MIN_BUDGET_PER_FACT)) for f, _ in scored_facts]

        # 第一轮：比例分配
        raw: list[tuple[Fact, float, float]] = [
            (f, s, total_token_budget * (s / total_mas)) for f, s in scored_facts
        ]

        # 钳位到 [MIN, MAX]
        clamped: list[tuple[Fact, int]] = []
        leftover = 0.0
        for f, s, tokens in raw:
            if tokens < self.MIN_BUDGET_PER_FACT:
                clamped.append((f, self.MIN_BUDGET_PER_FACT))
                leftover += tokens - self.MIN_BUDGET_PER_FACT
            elif tokens > self.MAX_BUDGET_PER_FACT:
                clamped.append((f, self.MAX_BUDGET_PER_FACT))
                leftover += tokens - self.MAX_BUDGET_PER_FACT
            else:
                clamped.append((f, int(tokens)))

        # 如果有剩余，分配给未封顶的 Fact
        if leftover > 0:
            unclamped = [(f, s) for (f, s), (_, t) in zip(scored_facts, clamped)
                         if self.MIN_BUDGET_PER_FACT < t < self.MAX_BUDGET_PER_FACT]
            unclamped_mas = sum(s for _, s in unclamped)
            if unclamped_mas > 0:
                for i in range(len(clamped)):
                    if self.MIN_BUDGET_PER_FACT < clamped[i][1] < self.MAX_BUDGET_PER_FACT:
                        extra = int(leftover * (scored_facts[i][1] / unclamped_mas))
                        clamped[i] = (clamped[i][0], min(clamped[i][1] + extra, self.MAX_BUDGET_PER_FACT))

        return clamped
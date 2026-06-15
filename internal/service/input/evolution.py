"""
演化决策模块。
==============

本模块负责调用 LLM 对输入信息执行演化决策。

职责：
  1. 接收新提取的事实和上下文信息（目标 MemBox 及其中已有的事实）。
  2. 调用 LLM 判断新事实与已有记忆的关系：ADD（新增）/ UPDATE（更新）/ MERGE（合并）/ LINK（关联）/ NOOP（无操作）。
  3. 根据决策结果，协调执行相应的记忆存储操作。

演化决策的意义：
  - 记忆系统不是简单地将所有事实堆砌在一起，而是需要判断信息是否冗余、
    是否与已有记忆冲突，从而决定是新增、替换、合并还是关联。
  - 保守策略以 ADD 为默认，避免过度合并导致信息丢失。

注意：当前本模块是架构占位层，实际的演化决策逻辑委托给
  LLMClient.evolution_decision() 方法执行，并由 MemoryPipeline.process_turn() 协调。
具体决策逻辑参见 internal.infrastructure.models.llm.llm.LLMClient
和相关的 Prompt 模板 EVOLUTION_DECISION_SYSTEM。
"""

# 这个文件负责调用之前的 LLM 模型来思考 对这套记忆执行什么操作
# 实际演化决策由 LLMClient.evolution_decision() 和 MemoryPipeline.process_turn() 协调执行

# AAAI Long-Term Memory System

一个面向长时记忆与多轮对话的检索增强管线：把对话流转为带元数据、标签和图结构的事实节点，按"图随机游走 + 多路融合 + 多智能体打分 + 上下文预算分配"组织最相关的记忆，再交由 LLM 合成回答。当前在 LongMemEval 公开评测集上做端到端验证。

## 特性

- **事实级写入**：LLM 将每条消息抽取为多个 `Fact`，自动填充 `Person/Object/Location/Event/Organization/Preference/HappendTime/MentionedTime` 等元数据，并通过 Trigram + Embedding + RRF 把候选实体合并为 Tag 列表。
- **多路混合召回**：BM25 / Trigram / 向量 / Tag 四路独立打分，min-max 归一化后加权融合（λ 可调），同时按时间窗与关键字两种入口并行召回。
- **图随机游走扩展**：基于「语义项 + 记忆项 + 结构项」的转移概率公式，在 Fact-Edge 图上从 seed 向邻域扩展，转移概率低于阈值时停止。
  ```
  P(u|v,q,t) = 1/Z_v · Γ(v,u,t) · exp(
      λ_sem · cos(e_q, e_u)
    + λ_mem · R(u,t)
    + λ_struct · [ln ω(v,u) − α · ln deg(u)]
  )
  ```
- **冲突解决**：写入与召回阶段都允许 LLM 标记 `conflict=true`，在边或 Fact 元数据上保留版本历史，避免硬覆盖。
- **拓扑感知遗忘（Ebbinghaus + 中心度）**：每次写入后触发遗忘周期，按 `topology_decay_lambda · recency` 降级 Fact，分 L0–L3 四个 tier。
- **MAS 多智能体打分**：semantic_match / edge_weight / recency / tier_boost / activation_history 五维加权综合得分。
- **CBA 上下文预算分配**：按 MAS 排序后做 token 预算分配，控制送给 LLM 的最终上下文长度（默认 4000 tokens）。
- **可观测**：DebugCollector 一键 dump 单题的 12 个阶段（forgetting / query_embed / keywords / entity_recall / fact_recall_per_path / fact_recall_fused / graph_walk / contradiction / mas_scored / cba_budget / final_context / llm_answer）。Token 用量按进程启动时间分文件持久化到 `TokenLog/`。
- **LongMemEval 端到端跑分**：自带评测 + 打分脚本，并发、断点续跑、单题超时、错误隔离。

## 仓库结构

```
.
├── main.py                          # MemoryRetrievalPipeline + CLI 入口
├── pyproject.toml                   # uv 包描述与依赖
├── internal/
│   ├── config/settings.py           # pydantic-settings；从 .env 读取
│   ├── domain/
│   │   ├── model/                   # Fact / Edge / Entity / MemBox / Tag / Tier / Buffer
│   │   └── services/
│   │       ├── mas_manager.py       # 多智能体加权打分
│   │       └── context_budget_allocator.py
│   ├── infra/
│   │   ├── database/sqlite.py       # aiosqlite + sqlite-vec
│   │   ├── repositories/            # Fact / Edge / Entity / MemBox 仓储
│   │   └── models/
│   │       ├── llm/llm.py           # 主 LLM 客户端（OpenAI 协议）
│   │       ├── embedding/embedding.py
│   │       └── judge/judge.py       # 评测打分客户端（Anthropic 协议）
│   ├── service/
│   │   ├── input/                   # 写入：filter / evolution / write_service
│   │   ├── retrieve/                # 召回：recall / manager / cba
│   │   └── forget/forget.py         # 遗忘衰减
│   └── util/
│       ├── api_retry.py             # 信号量 + 指数退避
│       ├── rrf.py                   # Reciprocal Rank Fusion
│       ├── token_tracker.py         # 全局 token 用量计数 + JSONL 日志
│       └── debug_collector.py       # 单题全流程 dump
├── scripts/
│   ├── evaluate_longmemeval.py      # LongMemEval 评测运行器（断点续跑）
│   ├── score_longmemeval.py         # Judge LLM 打分（按 question_type 分桶）
│   ├── debug_one_question.py        # 单题完整 dump
│   └── smoke_*.py                   # 模块级冒烟测试
└── LongMemEval/                     # 数据集占位（data/ 不入仓，见下文）
```

## 快速开始

### 1. 环境与依赖

需要 Python 3.14（`pyproject.toml` 里硬性要求）。推荐用 [uv](https://github.com/astral-sh/uv)：

```bash
uv sync                # 创建 .venv 并安装锁定版本依赖
```

主要依赖：`aiosqlite`、`sqlite-vec`、`anthropic`、`openai`、`numpy`、`pydantic-settings`、`tqdm`、`json-repair`、`volcengine-python-sdk[ark]`。

### 2. 配置凭证

```bash
cp .env.example .env
# 编辑 .env，填入真实 API key
```

需要三类凭证：
- `deepseek_api_key` / `deepseek_base_url` / `deepseek_model`：主 LLM（兼容 OpenAI Chat Completions 协议的任意网关）。
- `ark_api_key`：火山引擎 Ark 嵌入 API。
- `judge_api_key` / `judge_base_url` / `judge_model`：评测打分模型（兼容 Anthropic Messages 协议的任意网关）。

也可以通过环境变量直接注入（pydantic-settings 自动读取）。

### 3. 单条消息读写

```bash
PYTHONPATH=. uv run python main.py ingest "明天上午 10 点和张总开会，地点在公司 21 楼会议室。"
PYTHONPATH=. uv run python main.py answer "我什么时候和张总有会？"
```

CLI 默认把数据落到 `memory.db`（SQLite，含 sqlite-vec 向量扩展）。

## LongMemEval 评测

### 准备数据集

数据集没有纳入仓库（约 2.9 GB）。从官方仓库下载并放到 `LongMemEval/data/`：

```bash
# 参考：https://github.com/xiaowu0162/LongMemEval
git clone https://github.com/xiaowu0162/LongMemEval.git /tmp/lme
cp /tmp/lme/data/longmemeval_oracle.json LongMemEval/data/
```

至少需要 `longmemeval_oracle.json`（也可换成 `longmemeval_s.json` 等）。

### 跑评测

```bash
PYTHONPATH=. uv run python scripts/evaluate_longmemeval.py \
    --data longmemeval_oracle.json \
    --output /home/manjaro/tmp/eval_run/results.jsonl \
    --max 50 \
    --concurrency 4 \
    --per-item-timeout 1800
```

特性：
- **断点续跑**：output 文件中已有的 `question_id` 自动跳过（append 模式）。
- **单题超时**：默认 720 s，haystack 大题建议提到 1800 s。
- **错误隔离**：单题异常或超时只写入 `errors.jsonl`，不影响其余题目。
- **资源清理**：每题独立 SQLite 文件，结束后立即关闭并删除。
- **限流**：题目级 `--concurrency` + LLM 客户端层信号量（`settings.llm_max_concurrency` 等）双重控制。

### 打分

```bash
PYTHONPATH=. uv run python scripts/score_longmemeval.py \
    --hyp /home/manjaro/tmp/eval_run/results.jsonl \
    --ref LongMemEval/data/longmemeval_oracle.json \
    --concurrency 8
```

按官方模板分别处理 `temporal-reasoning`、`knowledge-update`、`single-session-preference`、`abstention`、默认五种打分模板，输出整体准确率与分桶准确率。

### 单题调试

```bash
PYTHONPATH=. uv run python scripts/debug_one_question.py \
    --qid 0bb5a684 \
    --data longmemeval_oracle.json \
    --out /home/manjaro/tmp/debug_0bb5a684.json
```

dump 出 12 个阶段的输入/输出，方便用 `jq` 定位"召回挂了"还是"LLM 拒答"。

## 核心算法参数

`internal/config/settings.py` 里关键的可调参数（默认值已在仓库内）：

| 模块 | 参数 | 含义 |
|---|---|---|
| 召回融合 | `retrieve_lambda_bm25` / `vec` / `tag` / `trigram` | 4 路加权融合系数 |
| 召回 top-K | `retrieve_top_k_fact_bm25/vec/tag/trigram/entity` | 每路截断深度 |
| 图游走 | `retrieve_lambda_sem/mem/struct`、`retrieve_alpha`、`retrieve_max_graph_depth` | 转移概率三项与度惩罚、最大跳数 |
| 遗忘 | `forgetting_fact_decay_rate`、`forgetting_tier_thresholds`、`retrieve_tau_days`、`retrieve_eta` | 衰减速率、tier 阈值、记忆项参数 |
| MAS | `mas_weights` | 五维子项权重 |
| 预算 | `retrieve_total_token_budget` | LLM 上下文 token 预算 |
| 限流 | `llm_max_concurrency`、`embedding_max_concurrency`、`api_max_retries` | 客户端层并发与重试 |

## Token 用量与日志

每次进程启动会在 `TokenLog/YYYY-MM-DD/{llm,embedding,judge}_HHMMSS.jsonl` 记录每次调用与运行结束的汇总（`type=summary`）。评测脚本结束时会打印 `--- Token usage ---` 表格。

## 已知限制

- 数据集与 SQLite 文件不入仓，需要本地准备。
- 主 LLM / 嵌入 / Judge 三类客户端均假设网关协议固定（OpenAI / Volcengine Ark / Anthropic）；换网关需要小改 `internal/infra/models/`。
- LongMemEval 全量 500 题在不同 question_type 上表现不均匀；temporal-reasoning 和 single-session-preference 仍是当前的薄弱项。

## 参考

- WorkFlow 设计文档：仓库根目录的 `WorkFlow` 文件
- LongMemEval：https://github.com/xiaowu0162/LongMemEval

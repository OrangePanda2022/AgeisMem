"""
全局应用配置模块 (Pydantic Settings)。

基于 pydantic-settings 管理应用程序的全部配置项，支持从 .env 文件自动加载环境变量。
配置涵盖数据库连接、LLM API 密钥、嵌入向量维度、遗忘服务参数以及 MAS 权重等。
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用全局配置，自动从 .env 文件和环境变量中加载。"""
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ---- LLM API 密钥与模型配置 ----
    # 通过 .env 或环境变量注入；默认空字符串避免源码携带凭证。
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # ---- Judge LLM ----
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_model: str = ""

    # ---- Embedding 模型 ----
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dim: int = 2048

    # ---- 外部 API 并发与重试 ----
    # 客户端层全局信号量限流，避免大并发评测撞限速。
    llm_max_concurrency: int = 48
    embedding_max_concurrency: int = 16
    api_max_retries: int = 8
    api_retry_base_delay: float = 2.0  # 指数退避基数（秒），8次重试最大延迟~256s，足以跨过RPM窗口
    # 单次调用硬超时（秒）。两层兜底：
    #   1. SDK 层 httpx timeout（连接/读取/写入），由 socket close 强制中断
    #   2. with_retry 外层 asyncio.wait_for（防止 SDK timeout 在某些网关下不触发）
    # 经验：FACT_EXTRACTION 系 LLM 一般 5-15s，超过 30s 基本是网关挂了。
    llm_call_timeout_s: float = 45.0
    embedding_call_timeout_s: float = 20.0
    judge_call_timeout_s: float = 30.0

    # ---- ForgettingService 遗忘服务参数 ----
    topology_decay_lambda: float = 1.0
    forgetting_membox_decay_rate: float = 0.05
    forgetting_fact_decay_rate: float = 0.1
    forgetting_tier_thresholds: dict = {
        "L0_L1": 0.7,
        "L1_L2": 0.4,
        "L2_L3": 0.15,
    }

    # ---- Graph random walk retrieval parameters ----
    retrieve_top_k_entity: int = 20
    retrieve_top_k_fact_vec: int = 10
    retrieve_top_k_fact_tag: int = 20
    retrieve_max_graph_depth: int = 4
    retrieve_graph_walk_threshold: float = 0.01
    retrieve_lambda_sem: float = 1.0
    retrieve_lambda_mem: float = 0.5
    retrieve_lambda_struct: float = 0.8
    retrieve_alpha: float = 0.3
    retrieve_tau_days: int = 365
    retrieve_eta: float = 0.5
    retrieve_total_token_budget: int = 6000

    # ---- Fact 召回 4 路加权融合（BM25 + Trigram + Vec + Tag）----
    retrieve_top_k_fact_bm25: int = 20
    retrieve_top_k_fact_trigram: int = 20
    retrieve_lambda_bm25: float = 0.6
    retrieve_lambda_vec: float = 1.5
    retrieve_lambda_tag: float = 0.4
    retrieve_lambda_trigram: float = 0.2

    # ---- MASManager 多智能体系统权重 ----
    mas_weights: dict = {
        "semantic_match": 0.45,
        "edge_weight": 0.20,
        "recency": 0.15,
        "tier_boost": 0.05,
        "activation_history": 0.15,
    }

    # ---- Iterative retrieval (sufficiency-check + query-expansion) ----
    iterative_retrieval_enabled: bool = True
    iterative_retrieval_max_rounds: int = 3

    # ---- Multi-agent debate mode ----
    debate_mode_enabled: bool = False
    debate_mode_specialists: int = 3

    # ---- Speculative retrieval ----
    speculative_retrieval_enabled: bool = True
    speculative_confidence_threshold: float = 0.8


# 全局单例配置实例，供各模块直接导入使用
settings = Settings()

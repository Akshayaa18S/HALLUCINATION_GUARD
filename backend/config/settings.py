"""
Centralized application configuration.

All environment-driven values are defined here and nowhere else, so the
rest of the codebase depends on `settings`, not on `os.environ` directly.
This makes every later phase (LLM service, retrieval, verification, etc.)
configurable without touching business logic.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "hallucination-detector"
    app_env: str = "development"
    debug: bool = True

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./database/app.db"

    # --- Auth ---
    # Override via .env in any real deployment - this default is only for
    # local/dev use out of the box.
    secret_key: str = "dev-only-change-this-secret-key-please"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days


    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # --- LLM (wired up starting Phase 3) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_top_k: int = 1
    llm_top_p: float = 1.0

    # --- Performance & Caching ---
    enable_intermediate_cache: bool = True

    # --- Retrieval (Phase 7) ---

    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    wikipedia_lang: str = "en"
    # Wikimedia's API returns 403 to requests without an identifying
    # User-Agent (see https://w.wiki/4wJS) - httpx's default UA gets
    # blocked, so every request needs this set explicitly. Update the
    # contact info if you deploy this somewhere Wikimedia could reach out
    # about usage.
    wikipedia_user_agent: str = (
        "HallucinationDetectorBackend/1.0 "
        "(https://github.com/example/hallucination-detector; contact@example.com) "
        "python-httpx"
    )
    retrieval_timeout_seconds: float = 10.0
    retrieval_top_k: int = 3
    min_evidence_score: float = 0.10
    cache_dir: str = "./cache"
    cache_ttl_seconds: int = 86400  # 24h


    # Configurable decision threshold for MultiHaluDet model probability
    # (optimal tuned threshold is 0.20 based on validation ROC-AUC 0.718 & F1 0.7117)
    multihaludet_decision_threshold: float = 0.20

    # Configurable Evidence Selector Ranking Weights
    ranking_semantic_weight: float = 0.35
    ranking_entity_weight: float = 0.25
    ranking_relation_weight: float = 0.25
    ranking_coverage_weight: float = 0.15

    # Path to a local FEVER dataset (jsonl of {claim, label, evidence_wiki_pages})
    # if you have one downloaded. Left unset -> FEVER retrieval returns no
    # evidence rather than failing, per Phase 7's graceful-degradation design.
    fever_dataset_path: str | None = None

    # --- Job runner / dev-mode simulation ---
    # (Referenced by services/pipeline_service.py; these were previously
    # unset and would raise AttributeError at PipelineService construction
    # time - added here to actually fix that.)
    DEV_MODE: bool = False
    JOB_MAX_RETRIES: int = 3
    DELAY_SIMULATION_ENABLED: bool = False
    STAGE_DELAY_MIN_MS: int = 100
    STAGE_DELAY_MAX_MS: int = 400

    # --- MultiHaluDet branch (hidden-state trajectory probing) ---
    # Local HF causal LM used for BOTH generation and hidden-state/logit
    # extraction, replacing the Ollama HTTP call for this pipeline. Ollama's
    # REST API cannot expose per-layer hidden states, which the MultiHaluDet
    # method requires, so the base-paper branch needs an in-process model.
    multihaludet_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    multihaludet_device: str = "cuda"  # "cpu" | "cuda" | "mps"
    multihaludet_dtype: str = "float16"  # "float32" | "float16" | "bfloat16"
    multihaludet_max_new_tokens: int = 256

    # Dynamic/multi-depth layer sampling: how many transformer layers
    # (out of the model's total) are probed. Evenly spaced across
    # early/middle/late depth, always including the first and last layer.
    multihaludet_num_sampled_layers: int = 6

    # Multi-scale attention: pooling window sizes applied to the per-layer
    # generation-step sequence before combining via learned gating.
    multihaludet_attention_scales: list[int] = [1, 2, 4]

    # Layer-weighted Transformer encoder dimensions.
    multihaludet_encoder_dim: int = 256
    multihaludet_encoder_heads: int = 4
    multihaludet_encoder_layers: int = 2

    # Global feature branch (top-k token probs, entropy, layer-norm
    # trajectory stats, etc.)
    multihaludet_global_top_k: int = 5
    multihaludet_global_hidden_dim: int = 64

    # Ensemble meta-learner: number of base-learner heads trained
    # out-of-fold on the fused deep-feature vector (paper: 4 stages,
    # last of which is OOF feature generation + learned meta-learning).
    multihaludet_ensemble_members: int = 5

    # Optional path to a trained checkpoint (produced by
    # multihaludet/training/train.py). If unset or missing, the model runs
    # with its randomly-initialized weights - architecturally faithful to
    # the paper, but NOT a trained hallucination detector. This is
    # surfaced explicitly in the API response's stage metadata so the
    # distinction is never silently hidden from a caller.
    multihaludet_checkpoint_path: str | None = "./backend/multihaludet/checkpoints/multihaludet.pt"

    # Configurable confidence calibration weights
    confidence_weights: dict[str, float] = {
        "ensemble": 0.40,
        "margin": 0.35,
        "evidence": 0.25,
    }

    # --- Dual-signal fusion (internal MultiHaluDet score x external RAG
    # evidence-verification score) ---
    # Weight given to the internal (MultiHaluDet) signal in the final
    # fused hallucination probability; the external RAG signal gets
    # (1 - fusion_internal_weight). Configured via FUSION_INTERNAL_WEIGHT env var.
    fusion_internal_weight: float = 0.70

    @property
    def log_dir_path(self) -> Path:
        path = Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir_path(self) -> Path:
        path = Path(self.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - avoids re-parsing .env on every call."""
    return Settings()


settings = get_settings()

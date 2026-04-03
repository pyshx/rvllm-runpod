from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_bool(name: str, default: bool = False) -> bool:
    value = _get_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerlessConfig:
    model_id: str | None
    model_target: str
    served_model_name: str
    tokenizer_id: str | None
    rvllm_port: int
    max_concurrency: int
    request_timeout: int
    ready_timeout: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    max_num_seqs: int
    rust_log: str
    disable_telemetry: bool
    hf_home: str
    hf_hub_cache: str

    @classmethod
    def from_env(cls) -> "ServerlessConfig":
        model_id = _get_env("MODEL_ID")
        baked_model_dir = _get_env("MODEL_DIR", "/models/default")
        explicit_target = _get_env("MODEL_TARGET")
        baked_model_available = False
        if baked_model_dir:
            baked_path = Path(baked_model_dir)
            baked_model_available = baked_path.is_dir() and any(baked_path.iterdir())
        model_target = explicit_target or (baked_model_dir if baked_model_available else None) or model_id or baked_model_dir
        served_model_name = _get_env("SERVED_MODEL_NAME") or model_id or model_target

        return cls(
            model_id=model_id,
            model_target=model_target,
            served_model_name=served_model_name,
            tokenizer_id=_get_env("TOKENIZER_ID"),
            rvllm_port=int(_get_env("RVLLM_PORT", "8000")),
            max_concurrency=int(_get_env("MAX_CONCURRENCY", "30")),
            request_timeout=int(_get_env("REQUEST_TIMEOUT", "600")),
            ready_timeout=int(_get_env("SERVER_READY_TIMEOUT", "900")),
            dtype=_get_env("DTYPE", "auto") or "auto",
            max_model_len=int(_get_env("MAX_MODEL_LEN", "2048")),
            gpu_memory_utilization=float(_get_env("GPU_MEMORY_UTILIZATION", "0.9")),
            tensor_parallel_size=int(_get_env("TENSOR_PARALLEL_SIZE", "1")),
            max_num_seqs=int(_get_env("MAX_NUM_SEQS", "256")),
            rust_log=_get_env("RUST_LOG", "info") or "info",
            disable_telemetry=_get_bool("DISABLE_TELEMETRY", False),
            hf_home=_get_env("HF_HOME", "/runpod-volume/huggingface") or "/runpod-volume/huggingface",
            hf_hub_cache=_get_env(
                "HUGGINGFACE_HUB_CACHE",
                f"{_get_env('HF_HOME', '/runpod-volume/huggingface') or '/runpod-volume/huggingface'}/hub",
            )
            or "/runpod-volume/huggingface/hub",
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.rvllm_port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def launch_command(self) -> list[str]:
        command = [
            "rvllm",
            "serve",
            "--model",
            self.model_target,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.rvllm_port),
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--log-level",
            self.rust_log,
        ]
        if self.tokenizer_id:
            command.extend(["--tokenizer", self.tokenizer_id])
        if self.disable_telemetry:
            command.append("--disable-telemetry")
        return command

    def launch_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("HF_HOME", self.hf_home)
        env.setdefault("HUGGINGFACE_HUB_CACHE", self.hf_hub_cache)
        env.setdefault("RUST_LOG", self.rust_log)
        return env

    def ensure_cache_dirs(self) -> None:
        Path(self.hf_home).mkdir(parents=True, exist_ok=True)
        Path(self.hf_hub_cache).mkdir(parents=True, exist_ok=True)

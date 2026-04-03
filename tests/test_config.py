from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ServerlessConfig, _get_bool, _get_env


# --- helpers ---


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all config-relevant env vars before each test."""
    for key in [
        "MODEL_ID", "MODEL_DIR", "MODEL_TARGET", "SERVED_MODEL_NAME",
        "TOKENIZER_ID", "RVLLM_PORT", "MAX_CONCURRENCY", "REQUEST_TIMEOUT",
        "SERVER_READY_TIMEOUT", "DTYPE", "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION", "TENSOR_PARALLEL_SIZE", "MAX_NUM_SEQS",
        "RUST_LOG", "DISABLE_TELEMETRY", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
    ]:
        monkeypatch.delenv(key, raising=False)


# --- _get_env ---


class TestGetEnv:
    def test_returns_default_when_unset(self):
        assert _get_env("__NONEXISTENT__", "fallback") == "fallback"

    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("__TEST_VAR__", "hello")
        assert _get_env("__TEST_VAR__") == "hello"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("__TEST_VAR__", "  spaced  ")
        assert _get_env("__TEST_VAR__") == "spaced"

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("__TEST_VAR__", "   ")
        assert _get_env("__TEST_VAR__", "fallback") == "fallback"

    def test_none_default(self):
        assert _get_env("__NONEXISTENT__") is None


# --- _get_bool ---


class TestGetBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("__BOOL__", value)
        assert _get_bool("__BOOL__") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "anything"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("__BOOL__", value)
        assert _get_bool("__BOOL__") is False

    def test_unset_returns_default_false(self):
        assert _get_bool("__NONEXISTENT__") is False

    def test_unset_returns_default_true(self):
        assert _get_bool("__NONEXISTENT__", default=True) is True


# --- ServerlessConfig.from_env ---


class TestServerlessConfigFromEnv:
    def test_generic_model_defaults(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        config = ServerlessConfig.from_env()
        assert config.model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert config.model_target == "Qwen/Qwen2.5-7B-Instruct"
        assert config.served_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_explicit_target_overrides_model_id(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        monkeypatch.setenv("MODEL_TARGET", "/models/custom")
        config = ServerlessConfig.from_env()
        assert config.model_target == "/models/custom"

    def test_baked_model_target_differs_from_public_name(self, monkeypatch):
        monkeypatch.setenv("MODEL_TARGET", "/models/default")
        monkeypatch.setenv("SERVED_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        config = ServerlessConfig.from_env()
        assert config.model_target == "/models/default"
        assert config.served_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_existing_baked_directory_becomes_target(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text("{}", encoding="utf-8")
            monkeypatch.setenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
            monkeypatch.setenv("MODEL_DIR", tmpdir)
            config = ServerlessConfig.from_env()
            assert config.model_target == tmpdir

    def test_empty_baked_directory_falls_back_to_model_id(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            # empty dir — no files inside
            monkeypatch.setenv("MODEL_ID", "meta-llama/Llama-3.1-8B")
            monkeypatch.setenv("MODEL_DIR", tmpdir)
            config = ServerlessConfig.from_env()
            assert config.model_target == "meta-llama/Llama-3.1-8B"

    def test_nonexistent_baked_directory_falls_back(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "meta-llama/Llama-3.1-8B")
        monkeypatch.setenv("MODEL_DIR", "/no/such/path")
        config = ServerlessConfig.from_env()
        assert config.model_target == "meta-llama/Llama-3.1-8B"

    def test_numeric_overrides(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("RVLLM_PORT", "9000")
        monkeypatch.setenv("MAX_CONCURRENCY", "10")
        monkeypatch.setenv("REQUEST_TIMEOUT", "300")
        monkeypatch.setenv("SERVER_READY_TIMEOUT", "120")
        monkeypatch.setenv("MAX_MODEL_LEN", "4096")
        monkeypatch.setenv("GPU_MEMORY_UTILIZATION", "0.85")
        monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "2")
        monkeypatch.setenv("MAX_NUM_SEQS", "64")
        config = ServerlessConfig.from_env()
        assert config.rvllm_port == 9000
        assert config.max_concurrency == 10
        assert config.request_timeout == 300
        assert config.ready_timeout == 120
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.85
        assert config.tensor_parallel_size == 2
        assert config.max_num_seqs == 64

    def test_dtype_default_is_auto(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        config = ServerlessConfig.from_env()
        assert config.dtype == "auto"

    def test_dtype_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("DTYPE", "half")
        config = ServerlessConfig.from_env()
        assert config.dtype == "half"

    def test_disable_telemetry(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("DISABLE_TELEMETRY", "true")
        config = ServerlessConfig.from_env()
        assert config.disable_telemetry is True

    def test_hf_cache_paths_default(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        config = ServerlessConfig.from_env()
        assert config.hf_home == "/runpod-volume/huggingface"
        assert config.hf_hub_cache == "/runpod-volume/huggingface/hub"

    def test_hf_cache_paths_custom(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("HF_HOME", "/custom/hf")
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/custom/hf/cache")
        config = ServerlessConfig.from_env()
        assert config.hf_home == "/custom/hf"
        assert config.hf_hub_cache == "/custom/hf/cache"

    def test_no_model_id_uses_model_dir_as_fallback(self, monkeypatch):
        # no MODEL_ID, no MODEL_TARGET, no baked model — falls back to MODEL_DIR default
        config = ServerlessConfig.from_env()
        assert config.model_target == "/models/default"


# --- properties and methods ---


class TestServerlessConfigProperties:
    @pytest.fixture()
    def config(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        return ServerlessConfig.from_env()

    def test_base_url(self, config):
        assert config.base_url == "http://127.0.0.1:8000"

    def test_health_url(self, config):
        assert config.health_url == "http://127.0.0.1:8000/health"

    def test_custom_port_in_urls(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("RVLLM_PORT", "9999")
        config = ServerlessConfig.from_env()
        assert config.base_url == "http://127.0.0.1:9999"
        assert config.health_url == "http://127.0.0.1:9999/health"


class TestLaunchCommand:
    def test_contains_required_flags(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        monkeypatch.setenv("DTYPE", "half")
        config = ServerlessConfig.from_env()
        cmd = config.launch_command()
        assert cmd[0] == "rvllm"
        assert cmd[1] == "serve"
        assert "--model" in cmd
        assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
        assert "--dtype" in cmd
        assert "half" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd

    def test_includes_tokenizer_when_set(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("TOKENIZER_ID", "test/tokenizer")
        config = ServerlessConfig.from_env()
        cmd = config.launch_command()
        assert "--tokenizer" in cmd
        assert "test/tokenizer" in cmd

    def test_excludes_tokenizer_when_unset(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        config = ServerlessConfig.from_env()
        cmd = config.launch_command()
        assert "--tokenizer" not in cmd

    def test_includes_disable_telemetry_flag(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("DISABLE_TELEMETRY", "1")
        config = ServerlessConfig.from_env()
        cmd = config.launch_command()
        assert "--disable-telemetry" in cmd

    def test_excludes_disable_telemetry_by_default(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        config = ServerlessConfig.from_env()
        cmd = config.launch_command()
        assert "--disable-telemetry" not in cmd


class TestLaunchEnv:
    def test_inherits_os_environ(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("SOME_OTHER_VAR", "preserved")
        config = ServerlessConfig.from_env()
        env = config.launch_env()
        assert env["SOME_OTHER_VAR"] == "preserved"

    def test_sets_hf_defaults(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
        config = ServerlessConfig.from_env()
        env = config.launch_env()
        assert env["HF_HOME"] == "/runpod-volume/huggingface"
        assert env["HUGGINGFACE_HUB_CACHE"] == "/runpod-volume/huggingface/hub"


class TestEnsureCacheDirs:
    def test_creates_directories(self, monkeypatch, tmp_path):
        hf_home = str(tmp_path / "hf")
        hf_cache = str(tmp_path / "hf" / "hub")
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", hf_cache)
        config = ServerlessConfig.from_env()
        config.ensure_cache_dirs()
        assert Path(hf_home).is_dir()
        assert Path(hf_cache).is_dir()

    def test_idempotent(self, monkeypatch, tmp_path):
        hf_home = str(tmp_path / "hf")
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("HF_HOME", hf_home)
        config = ServerlessConfig.from_env()
        config.ensure_cache_dirs()
        config.ensure_cache_dirs()  # second call should not raise
        assert Path(hf_home).is_dir()

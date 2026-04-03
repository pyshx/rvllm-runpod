from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from request_mapping import (
    RequestMappingError,
    build_proxy_request,
    rewrite_request_model,
    rewrite_response_models,
)

SERVED = "Qwen/Qwen2.5-7B-Instruct"
TARGET = "/models/default"


# --- build_proxy_request: chat inference ---


class TestBuildProxyRequestChat:
    def test_chat_messages_inferred(self):
        req = build_proxy_request(
            {"messages": [{"role": "user", "content": "hi"}]}, SERVED
        )
        assert req.path == "/v1/chat/completions"
        assert req.method == "POST"
        assert req.body["model"] == SERVED
        assert req.stream is False

    def test_chat_with_stream(self):
        req = build_proxy_request(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True}, SERVED
        )
        assert req.stream is True

    def test_chat_preserves_extra_fields(self):
        req = build_proxy_request(
            {"messages": [], "temperature": 0.5, "max_tokens": 100}, SERVED
        )
        assert req.body["temperature"] == 0.5
        assert req.body["max_tokens"] == 100

    def test_chat_does_not_overwrite_explicit_model(self):
        req = build_proxy_request(
            {"messages": [], "model": "custom/model"}, SERVED
        )
        assert req.body["model"] == "custom/model"


# --- build_proxy_request: text completion ---


class TestBuildProxyRequestCompletion:
    def test_prompt_inferred(self):
        req = build_proxy_request({"prompt": "hello"}, SERVED)
        assert req.path == "/v1/completions"
        assert req.body["model"] == SERVED

    def test_prompt_with_stream(self):
        req = build_proxy_request({"prompt": "hello", "stream": True}, SERVED)
        assert req.stream is True


# --- build_proxy_request: explicit path ---


class TestBuildProxyRequestExplicit:
    def test_get_models(self):
        req = build_proxy_request({"path": "/v1/models", "method": "GET"}, SERVED)
        assert req.method == "GET"
        assert req.path == "/v1/models"
        assert req.body is None
        assert req.stream is False

    def test_endpoint_alias(self):
        req = build_proxy_request(
            {"endpoint": "/v1/chat/completions", "body": {"messages": []}}, SERVED
        )
        assert req.path == "/v1/chat/completions"
        assert req.body["model"] == SERVED

    def test_path_takes_precedence_over_endpoint(self):
        req = build_proxy_request(
            {"path": "/v1/completions", "endpoint": "/v1/chat/completions"}, SERVED
        )
        assert req.path == "/v1/completions"

    def test_explicit_post_with_body(self):
        req = build_proxy_request(
            {
                "path": "/v1/chat/completions",
                "method": "POST",
                "body": {"messages": [], "stream": True},
            },
            SERVED,
        )
        assert req.stream is True
        assert req.body["model"] == SERVED

    def test_explicit_path_non_model_endpoint(self):
        req = build_proxy_request(
            {"path": "/v1/custom", "body": {"data": "value"}}, SERVED
        )
        # non-standard endpoint should not inject model
        assert "model" not in req.body

    def test_body_must_be_object(self):
        with pytest.raises(RequestMappingError, match="body must be an object"):
            build_proxy_request(
                {"path": "/v1/chat/completions", "body": "not a dict"}, SERVED
            )

    def test_default_method_is_post(self):
        req = build_proxy_request({"path": "/v1/completions"}, SERVED)
        assert req.method == "POST"

    def test_empty_body_gives_none(self):
        req = build_proxy_request({"path": "/v1/models"}, SERVED)
        assert req.body is None


# --- build_proxy_request: action shorthand ---


class TestBuildProxyRequestAction:
    def test_action_models(self):
        req = build_proxy_request({"action": "models"}, SERVED)
        assert req.path == "/v1/models"
        assert req.method == "GET"
        assert req.body is None


# --- build_proxy_request: errors ---


class TestBuildProxyRequestErrors:
    def test_unrecognized_payload_raises(self):
        with pytest.raises(RequestMappingError, match="could not infer route"):
            build_proxy_request({"temperature": 0.1}, SERVED)

    def test_non_dict_input_raises(self):
        with pytest.raises(RequestMappingError, match="must be a JSON object"):
            build_proxy_request("not a dict", SERVED)

    def test_list_input_raises(self):
        with pytest.raises(RequestMappingError, match="must be a JSON object"):
            build_proxy_request([1, 2, 3], SERVED)


# --- rewrite_request_model ---


class TestRewriteRequestModel:
    def test_rewrites_matching_model(self):
        result = rewrite_request_model(
            {"model": SERVED, "messages": []}, SERVED, TARGET
        )
        assert result["model"] == TARGET
        assert result["messages"] == []

    def test_rewrites_none_model(self):
        result = rewrite_request_model({"messages": []}, SERVED, TARGET)
        assert result["model"] == TARGET

    def test_preserves_foreign_model(self):
        result = rewrite_request_model(
            {"model": "other/model", "messages": []}, SERVED, TARGET
        )
        assert result["model"] == "other/model"

    def test_none_body_returns_none(self):
        assert rewrite_request_model(None, SERVED, TARGET) is None

    def test_does_not_mutate_input(self):
        original = {"model": SERVED, "messages": []}
        rewrite_request_model(original, SERVED, TARGET)
        assert original["model"] == SERVED


# --- rewrite_response_models ---


class TestRewriteResponseModels:
    def test_rewrites_top_level_model(self):
        result = rewrite_response_models({"model": TARGET}, SERVED, TARGET)
        assert result["model"] == SERVED

    def test_rewrites_nested_model_list(self):
        payload = {
            "data": [{"id": TARGET, "object": "model", "model": TARGET}],
        }
        result = rewrite_response_models(payload, SERVED, TARGET)
        assert result["data"][0]["id"] == SERVED
        assert result["data"][0]["model"] == SERVED

    def test_preserves_unrelated_values(self):
        payload = {"model": TARGET, "usage": {"prompt_tokens": 10}}
        result = rewrite_response_models(payload, SERVED, TARGET)
        assert result["usage"]["prompt_tokens"] == 10

    def test_leaves_non_target_model_alone(self):
        result = rewrite_response_models({"model": "other"}, SERVED, TARGET)
        assert result["model"] == "other"

    def test_handles_empty_dict(self):
        assert rewrite_response_models({}, SERVED, TARGET) == {}

    def test_handles_empty_list(self):
        assert rewrite_response_models([], SERVED, TARGET) == []

    def test_handles_scalar(self):
        assert rewrite_response_models("hello", SERVED, TARGET) == "hello"
        assert rewrite_response_models(42, SERVED, TARGET) == 42
        assert rewrite_response_models(None, SERVED, TARGET) is None

    def test_deeply_nested(self):
        payload = {"a": {"b": {"c": [{"model": TARGET}]}}}
        result = rewrite_response_models(payload, SERVED, TARGET)
        assert result["a"]["b"]["c"][0]["model"] == SERVED

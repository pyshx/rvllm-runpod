from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RequestMappingError(ValueError):
    pass


@dataclass(frozen=True)
class ProxyRequest:
    method: str
    path: str
    body: dict[str, Any] | None
    stream: bool


def build_proxy_request(job_input: dict[str, Any], served_model_name: str) -> ProxyRequest:
    if not isinstance(job_input, dict):
        raise RequestMappingError("job input must be a JSON object")

    if "path" in job_input or "endpoint" in job_input:
        path = str(job_input.get("path") or job_input.get("endpoint"))
        method = str(job_input.get("method", "POST")).upper()
        body = job_input.get("body")
        if body is not None and not isinstance(body, dict):
            raise RequestMappingError("body must be an object when provided")
        normalized_body = dict(body) if body else None
        if normalized_body and path in _MODEL_BODY_ENDPOINTS:
            normalized_body.setdefault("model", served_model_name)
        return ProxyRequest(
            method=method,
            path=path,
            body=normalized_body,
            stream=bool(normalized_body and normalized_body.get("stream")),
        )

    if "messages" in job_input:
        body = dict(job_input)
        body.setdefault("model", served_model_name)
        return ProxyRequest(
            method="POST",
            path="/v1/chat/completions",
            body=body,
            stream=bool(body.get("stream")),
        )

    if "prompt" in job_input:
        body = dict(job_input)
        body.setdefault("model", served_model_name)
        return ProxyRequest(
            method="POST",
            path="/v1/completions",
            body=body,
            stream=bool(body.get("stream")),
        )

    if job_input.get("action") == "models":
        return ProxyRequest(method="GET", path="/v1/models", body=None, stream=False)

    raise RequestMappingError(
        "could not infer route; provide messages, prompt, or explicit path/endpoint"
    )


def rewrite_request_model(body: dict[str, Any] | None, served_model_name: str, model_target: str) -> dict[str, Any] | None:
    if body is None:
        return None
    rewritten = dict(body)
    model = rewritten.get("model")
    if model is None or model == served_model_name:
        rewritten["model"] = model_target
    return rewritten


def rewrite_response_models(payload: Any, served_model_name: str, model_target: str) -> Any:
    if isinstance(payload, dict):
        rewritten = {}
        for key, value in payload.items():
            if key == "model" and value == model_target:
                rewritten[key] = served_model_name
                continue
            rewritten[key] = rewrite_response_models(value, served_model_name, model_target)
        if "id" in rewritten and rewritten.get("object") == "model" and rewritten["id"] == model_target:
            rewritten["id"] = served_model_name
        return rewritten
    if isinstance(payload, list):
        return [rewrite_response_models(item, served_model_name, model_target) for item in payload]
    return payload


_MODEL_BODY_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
}

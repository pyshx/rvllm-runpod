from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ServerlessConfig
from proxy import RvllmProxy
from request_mapping import ProxyRequest


@pytest.fixture()
def config(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "test/model")
    monkeypatch.setenv("SERVED_MODEL_NAME", "test/model")
    monkeypatch.setenv("RVLLM_PORT", "18000")
    return ServerlessConfig.from_env()


@pytest.fixture()
def proxy(config):
    return RvllmProxy(config)


def _chat_request(stream: bool = False) -> ProxyRequest:
    return ProxyRequest(
        method="POST",
        path="/v1/chat/completions",
        body={"model": "test/model", "messages": [{"role": "user", "content": "hi"}]},
        stream=stream,
    )


def _get_request() -> ProxyRequest:
    return ProxyRequest(method="GET", path="/v1/models", body=None, stream=False)


# --- non-streaming ---


class TestProxyNonStreaming:
    @respx.mock
    @pytest.mark.asyncio
    async def test_proxies_chat_completion(self, proxy):
        response_payload = {
            "model": "test/model",
            "choices": [{"message": {"content": "hello"}}],
        }
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response_payload)
        )

        results = []
        async for item in proxy.execute(_chat_request()):
            results.append(item)

        assert len(results) == 1
        assert results[0]["choices"][0]["message"]["content"] == "hello"

    @respx.mock
    @pytest.mark.asyncio
    async def test_proxies_get_models(self, proxy):
        response_payload = {
            "data": [{"id": "test/model", "object": "model"}],
        }
        respx.get("http://127.0.0.1:18000/v1/models").mock(
            return_value=httpx.Response(200, json=response_payload)
        )

        results = []
        async for item in proxy.execute(_get_request()):
            results.append(item)

        assert len(results) == 1
        assert results[0]["data"][0]["id"] == "test/model"

    @respx.mock
    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, proxy):
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in proxy.execute(_chat_request()):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_rewrites_model_in_response(self, proxy, config):
        # rvllm may return the local model target instead of the served name
        response_payload = {"model": config.model_target, "choices": []}
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=response_payload)
        )

        results = []
        async for item in proxy.execute(_chat_request()):
            results.append(item)

        assert results[0]["model"] == config.served_model_name


# --- streaming ---


def _sse_body(*chunks: dict, done: bool = True) -> str:
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines)


class TestProxyStreaming:
    @respx.mock
    @pytest.mark.asyncio
    async def test_streams_chunks(self, proxy):
        chunk1 = {"choices": [{"delta": {"content": "hel"}}], "model": "test/model"}
        chunk2 = {"choices": [{"delta": {"content": "lo"}}], "model": "test/model"}
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=_sse_body(chunk1, chunk2),
                headers={"content-type": "text/event-stream"},
            )
        )

        results = []
        async for item in proxy.execute(_chat_request(stream=True)):
            results.append(item)

        assert len(results) == 2
        assert results[0]["choices"][0]["delta"]["content"] == "hel"
        assert results[1]["choices"][0]["delta"]["content"] == "lo"

    @respx.mock
    @pytest.mark.asyncio
    async def test_skips_empty_lines(self, proxy):
        body = "\n\ndata: {\"ok\": true}\n\ndata: [DONE]\n\n"
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )

        results = []
        async for item in proxy.execute(_chat_request(stream=True)):
            results.append(item)

        assert len(results) == 1
        assert results[0]["ok"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self, proxy):
        body = "event: ping\ndata: {\"ok\": true}\n\ndata: [DONE]\n\n"
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )

        results = []
        async for item in proxy.execute(_chat_request(stream=True)):
            results.append(item)

        assert len(results) == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_skips_malformed_json_chunks(self, proxy):
        body = "data: not-json\ndata: {\"ok\": true}\n\ndata: [DONE]\n\n"
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )

        results = []
        async for item in proxy.execute(_chat_request(stream=True)):
            results.append(item)

        assert len(results) == 1
        assert results[0]["ok"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_raises_on_http_error(self, proxy):
        respx.post("http://127.0.0.1:18000/v1/chat/completions").mock(
            return_value=httpx.Response(502, text="Bad Gateway")
        )

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in proxy.execute(_chat_request(stream=True)):
                pass

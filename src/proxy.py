from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from config import ServerlessConfig
from request_mapping import ProxyRequest, rewrite_request_model, rewrite_response_models


class RvllmProxy:
    def __init__(self, config: ServerlessConfig):
        self.config = config

    async def execute(self, request: ProxyRequest) -> AsyncGenerator[dict[str, Any], None]:
        timeout = httpx.Timeout(self.config.request_timeout)
        async with httpx.AsyncClient(base_url=self.config.base_url, timeout=timeout) as client:
            if request.stream:
                async for chunk in self._stream_request(client, request):
                    yield chunk
                return

            response = await client.request(
                request.method,
                request.path,
                json=rewrite_request_model(
                    request.body,
                    self.config.served_model_name,
                    self.config.model_target,
                ),
            )
            response.raise_for_status()
            payload = response.json()
            yield rewrite_response_models(
                payload,
                self.config.served_model_name,
                self.config.model_target,
            )

    async def _stream_request(
        self, client: httpx.AsyncClient, request: ProxyRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        body = rewrite_request_model(
            request.body,
            self.config.served_model_name,
            self.config.model_target,
        )
        async with client.stream(request.method, request.path, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                payload = json.loads(data)
                yield rewrite_response_models(
                    payload,
                    self.config.served_model_name,
                    self.config.model_target,
                )

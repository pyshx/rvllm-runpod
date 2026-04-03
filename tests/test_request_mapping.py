import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from request_mapping import (
    RequestMappingError,
    build_proxy_request,
    rewrite_request_model,
    rewrite_response_models,
)


class RequestMappingTests(unittest.TestCase):
    def test_chat_payload_is_inferred(self):
        request = build_proxy_request(
            {"messages": [{"role": "user", "content": "hi"}]},
            "Qwen/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(request.path, "/v1/chat/completions")
        self.assertEqual(request.body["model"], "Qwen/Qwen2.5-7B-Instruct")

    def test_prompt_payload_is_inferred(self):
        request = build_proxy_request({"prompt": "hello"}, "demo/model")
        self.assertEqual(request.path, "/v1/completions")
        self.assertEqual(request.body["model"], "demo/model")

    def test_explicit_get_models_route(self):
        request = build_proxy_request({"path": "/v1/models", "method": "GET"}, "demo/model")
        self.assertEqual(request.method, "GET")
        self.assertIsNone(request.body)

    def test_request_model_rewrite_uses_local_target(self):
        rewritten = rewrite_request_model(
            {"model": "demo/model", "messages": []},
            "demo/model",
            "/models/default",
        )
        self.assertEqual(rewritten["model"], "/models/default")

    def test_response_model_rewrite_restores_public_name(self):
        payload = {
            "model": "/models/default",
            "data": [{"id": "/models/default", "object": "model"}],
        }
        rewritten = rewrite_response_models(payload, "demo/model", "/models/default")
        self.assertEqual(rewritten["model"], "demo/model")
        self.assertEqual(rewritten["data"][0]["id"], "demo/model")

    def test_invalid_payload_raises(self):
        with self.assertRaises(RequestMappingError):
            build_proxy_request({"temperature": 0.1}, "demo/model")


if __name__ == "__main__":
    unittest.main()

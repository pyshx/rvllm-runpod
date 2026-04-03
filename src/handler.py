from __future__ import annotations

import multiprocessing
import sys
import traceback

import runpod
from runpod import RunPodLogger

from config import ServerlessConfig
from proxy import RvllmProxy
from request_mapping import RequestMappingError, build_proxy_request
from server_launcher import RvllmServerLauncher

log = RunPodLogger()
config = ServerlessConfig.from_env()
launcher = RvllmServerLauncher(config)
proxy = RvllmProxy(config)


async def handler(job):
    try:
        launcher.assert_running()
        request = build_proxy_request(job["input"], config.served_model_name)
        async for item in proxy.execute(request):
            yield item
    except RequestMappingError as exc:
        yield {"error": str(exc), "type": "bad_request"}
    except Exception as exc:
        error_text = str(exc)
        log.error(f"rvLLM serverless request failed: {error_text}")
        log.error(traceback.format_exc())
        if "CUDA" in error_text or "cuda" in error_text:
            sys.exit(1)
        yield {"error": error_text, "type": "server_error"}


if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    try:
        launcher.start()
        log.info("rvLLM server started successfully")
    except Exception as exc:
        log.error(f"rvLLM startup failed: {exc}")
        log.error(traceback.format_exc())
        sys.exit(1)

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda _: config.max_concurrency,
            "return_aggregate_stream": True,
        }
    )

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request

from config import ServerlessConfig


class RvllmServerLauncher:
    def __init__(self, config: ServerlessConfig):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self.config.ensure_cache_dirs()
        self.process = subprocess.Popen(self.config.launch_command(), env=self.config.launch_env())
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        deadline = time.time() + self.config.ready_timeout
        last_error: str | None = None

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"rvLLM exited during startup with code {self.process.returncode}")

            try:
                with urllib.request.urlopen(self.config.health_url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError) as exc:
                last_error = str(exc)

            time.sleep(1)

        raise TimeoutError(f"rvLLM health check timed out: {last_error or 'no response'}")

    def assert_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"rvLLM process exited with code {self.process.returncode}")

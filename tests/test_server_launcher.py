from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ServerlessConfig
from server_launcher import RvllmServerLauncher


@pytest.fixture()
def config(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "test/model")
    monkeypatch.setenv("SERVER_READY_TIMEOUT", "2")
    return ServerlessConfig.from_env()


@pytest.fixture()
def launcher(config):
    return RvllmServerLauncher(config)


class TestAssertRunning:
    def test_no_process_does_not_raise(self, launcher):
        launcher.assert_running()

    def test_running_process_does_not_raise(self, launcher):
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        launcher.process = proc
        launcher.assert_running()

    def test_exited_process_raises(self, launcher):
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        launcher.process = proc
        with pytest.raises(RuntimeError, match="exited with code 1"):
            launcher.assert_running()

    def test_exited_with_zero_also_raises(self, launcher):
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        launcher.process = proc
        with pytest.raises(RuntimeError, match="exited with code 0"):
            launcher.assert_running()


class TestWaitUntilReady:
    @patch("server_launcher.urllib.request.urlopen")
    def test_returns_when_healthy(self, mock_urlopen, launcher):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        proc = MagicMock()
        proc.poll.return_value = None
        launcher.process = proc

        launcher.wait_until_ready()  # should not raise

    @patch("server_launcher.time.sleep")
    @patch("server_launcher.urllib.request.urlopen")
    def test_retries_until_healthy(self, mock_urlopen, mock_sleep, launcher):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        import urllib.error
        mock_urlopen.side_effect = [
            urllib.error.URLError("refused"),
            urllib.error.URLError("refused"),
            mock_response,
        ]

        proc = MagicMock()
        proc.poll.return_value = None
        launcher.process = proc

        launcher.wait_until_ready()
        assert mock_sleep.call_count == 2

    @patch("server_launcher.time.sleep")
    @patch("server_launcher.urllib.request.urlopen")
    @patch("server_launcher.time.time")
    def test_times_out(self, mock_time, mock_urlopen, mock_sleep, launcher):
        # time() returns values that exceed the deadline
        mock_time.side_effect = [0, 0, 100]

        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        proc = MagicMock()
        proc.poll.return_value = None
        launcher.process = proc

        with pytest.raises(TimeoutError, match="health check timed out"):
            launcher.wait_until_ready()

    @patch("server_launcher.urllib.request.urlopen")
    def test_raises_if_process_exits_during_startup(self, mock_urlopen, launcher):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("refused")

        proc = MagicMock()
        proc.poll.return_value = 137
        proc.returncode = 137
        launcher.process = proc

        with pytest.raises(RuntimeError, match="exited during startup with code 137"):
            launcher.wait_until_ready()


class TestStart:
    @patch.object(RvllmServerLauncher, "wait_until_ready")
    @patch("server_launcher.subprocess.Popen")
    def test_launches_process(self, mock_popen, mock_wait, launcher, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf" / "hub"))
        # re-create config with tmp paths
        launcher.config = ServerlessConfig.from_env()

        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        launcher.start()

        mock_popen.assert_called_once()
        args = mock_popen.call_args
        cmd = args[0][0]
        assert cmd[0] == "rvllm"
        assert cmd[1] == "serve"
        assert launcher.process is mock_proc
        mock_wait.assert_called_once()

    @patch.object(RvllmServerLauncher, "wait_until_ready")
    @patch("server_launcher.subprocess.Popen")
    def test_creates_cache_dirs(self, mock_popen, mock_wait, launcher, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        monkeypatch.setenv("MODEL_ID", "test/model")
        monkeypatch.setenv("HF_HOME", str(hf_home))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        launcher.config = ServerlessConfig.from_env()

        mock_popen.return_value = MagicMock()
        launcher.start()

        assert hf_home.is_dir()

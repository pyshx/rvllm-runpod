import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ServerlessConfig


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_generic_model_defaults(self):
        os.environ["MODEL_ID"] = "Qwen/Qwen2.5-7B-Instruct"
        config = ServerlessConfig.from_env()
        self.assertEqual(config.model_target, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(config.served_model_name, "Qwen/Qwen2.5-7B-Instruct")

    def test_baked_model_target_can_differ_from_public_name(self):
        os.environ["MODEL_TARGET"] = "/models/default"
        os.environ["SERVED_MODEL_NAME"] = "Qwen/Qwen2.5-7B-Instruct"
        config = ServerlessConfig.from_env()
        self.assertEqual(config.model_target, "/models/default")
        self.assertEqual(config.served_model_name, "Qwen/Qwen2.5-7B-Instruct")

    def test_existing_baked_directory_becomes_default_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text("{}", encoding="utf-8")
            os.environ["MODEL_ID"] = "Qwen/Qwen2.5-7B-Instruct"
            os.environ["MODEL_DIR"] = tmpdir
            config = ServerlessConfig.from_env()
            self.assertEqual(config.model_target, tmpdir)

    def test_launch_command_contains_expected_flags(self):
        os.environ["MODEL_ID"] = "meta-llama/Llama-3.1-8B-Instruct"
        os.environ["TOKENIZER_ID"] = "meta-llama/Llama-3.1-8B-Instruct"
        os.environ["DTYPE"] = "half"
        config = ServerlessConfig.from_env()
        command = config.launch_command()
        self.assertIn("--model", command)
        self.assertIn("meta-llama/Llama-3.1-8B-Instruct", command)
        self.assertIn("--tokenizer", command)
        self.assertIn("half", command)


if __name__ == "__main__":
    unittest.main()

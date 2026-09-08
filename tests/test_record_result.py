import importlib.util
import os
import subprocess
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(ROOT, "container", "record_result.py")
SPEC = importlib.util.spec_from_file_location("record_result", MODULE_PATH)
record_result = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record_result)


class TestResultLogParsing(unittest.TestCase):
    def test_residual_target_takes_precedence_over_base_parameter_time(self):
        log = """
  parameters : stage4.json (measured 108.692 s)
  target     : 106.630 s
RESULTS
  all laps      : ['114.130', '107.200']
  best warm lap : 107.200 s
"""
        result = record_result.parse_log(log)
        self.assertEqual(result["laps_s"], [114.130, 107.200])
        self.assertEqual(result["best_warm_lap_s"], 107.200)
        self.assertEqual(result["reference_s"], 106.630)

    def test_old_log_falls_back_to_parameter_time(self):
        log = """
  parameters : stage4.json (measured 108.692 s)
RESULTS
  all laps      : ['114.130', '108.538']
  best warm lap : 108.538 s
"""
        result = record_result.parse_log(log)
        self.assertEqual(result["reference_s"], 108.692)


class TestGitHelper(unittest.TestCase):
    def test_clean_repository_returns_empty_status_not_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(("git", "init", "-q", tmp), check=True)
            self.assertEqual(
                record_result.git("status", "--porcelain", cwd=tmp), ""
            )

    def test_failed_git_command_returns_none(self):
        self.assertIsNone(record_result.git("not-a-real-command"))


if __name__ == "__main__":
    unittest.main()

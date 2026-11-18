import tempfile
import unittest
from pathlib import Path

from compare_results import compare_results


class ResultComparisonTests(unittest.TestCase):
    def compare(self, first: str, second: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left, right = (Path(temporary) / name for name in ("gui.json", "cli.json"))
            left.write_text(first)
            right.write_text(second)
            compare_results(left, right)

    def test_object_key_order_does_not_change_results(self) -> None:
        self.compare(
            '{"status":true,"actions":["a","b"]}', '{"actions":["a","b"],"status":true}'
        )

    def test_action_order_changes_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ"):
            self.compare('["a","b"]', '["b","a"]')

    def test_boolean_and_number_are_distinct(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ"):
            self.compare('{"stable":true}', '{"stable":1}')

    def test_duplicate_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.compare('{"stable":false,"stable":true}', '{"stable":true}')

    def test_nonfinite_number_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.compare('{"value":NaN}', '{"value":NaN}')

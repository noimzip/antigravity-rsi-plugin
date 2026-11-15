"""Unit tests for sample tasklib module."""

from __future__ import annotations

import unittest

from src.tasklib import moving_average, normalize_tags


class TasklibTests(unittest.TestCase):
    def test_moving_average_basic(self) -> None:
        self.assertEqual(moving_average([1.0, 2.0, 3.0, 4.0], 2), [1.5, 2.5, 3.5])

    def test_moving_average_edge_cases(self) -> None:
        self.assertEqual(moving_average([], 2), [])
        self.assertEqual(moving_average([1.0], 3), [])
        with self.assertRaises(ValueError):
            moving_average([1.0, 2.0], 0)

    def test_normalize_tags(self) -> None:
        self.assertEqual(
            normalize_tags(["  Foo ", "bar", "FOO", "", "  Bar  ", "baz"]),
            ["foo", "bar", "baz"],
        )


if __name__ == "__main__":
    unittest.main()

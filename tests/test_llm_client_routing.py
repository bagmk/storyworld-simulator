#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.novel_writer.llm_client import LLMClient


class LLMClientRoutingTest(unittest.TestCase):
    def _make_client(self, model: str = "gpt-4o-mini", premium_model: str = "gpt-5-mini") -> LLMClient:
        client = object.__new__(LLMClient)
        client.model = model
        client.premium_model = premium_model
        client.budget_usd = 5.0
        client.temperature = 0.8
        client._spent_usd = 0.0
        client._usage_log = []
        return client

    def test_core_prose_purpose_keeps_gpt5(self) -> None:
        client = self._make_client()
        self.assertEqual(
            client._resolve_model(use_premium=True, purpose="prose_scene_gen"),
            "gpt-5-mini",
        )
        self.assertEqual(
            client._resolve_model(use_premium=True, purpose="prose_polish"),
            "gpt-5-mini",
        )

    def test_non_core_gpt5_premium_purpose_falls_back_to_cheap_model(self) -> None:
        client = self._make_client()
        self.assertEqual(
            client._resolve_model(use_premium=True, purpose="director_failure_analysis"),
            "gpt-4o-mini",
        )
        self.assertEqual(
            client._resolve_model(use_premium=True, purpose="scene_distillation"),
            "gpt-4o-mini",
        )

    def test_non_gpt5_premium_model_still_uses_premium(self) -> None:
        client = self._make_client(premium_model="gpt-4o")
        self.assertEqual(
            client._resolve_model(use_premium=True, purpose="quality_reviewer_llm_review"),
            "gpt-4o",
        )

    def test_non_premium_calls_always_use_cheap_model(self) -> None:
        client = self._make_client()
        self.assertEqual(
            client._resolve_model(use_premium=False, purpose="prose_scene_gen"),
            "gpt-4o-mini",
        )

    def test_gpt5_mini_usage_cost_is_recorded(self) -> None:
        client = self._make_client()
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=1000)
        )
        client._record_usage(response, "gpt-5-mini", "prose_scene_gen")
        self.assertAlmostEqual(client.spent_usd, 0.00225, places=8)
        summary = client.budget_summary()
        self.assertEqual(summary["call_count"], 1)
        self.assertEqual(summary["breakdown"][0]["model"], "gpt-5-mini")
        self.assertAlmostEqual(summary["breakdown"][0]["cost"], 0.00225, places=6)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import daily_pipeline


class DailyPipelineEpisodeResetTest(unittest.TestCase):
    def test_reset_story_state_from_episode_keeps_prior_episode_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            story_state_path = Path(tmpdir) / "story_state.json"
            backup_dir = Path(tmpdir) / "backup"
            story_state_path.write_text(
                json.dumps(
                    {
                        "last_completed_episode": "ep02_followup",
                        "arc_position": {"act": 2},
                        "character_states": {"kim_sumin": {"mood": "alert"}},
                        "active_clues": {"cipher": {"status": "open"}},
                        "episode_summaries": {
                            "ep01_intro": {"cycle_count": 2},
                            "ep02_followup": {"cycle_count": 1},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = daily_pipeline._reset_story_state_from_episode(
                story_state_path,
                "ep02_followup",
                backup_dir,
            )

            self.assertTrue(result["changed"])
            self.assertEqual(result["removed_episode_keys"], ["ep02_followup"])
            self.assertEqual(result["remaining_episode_keys"], ["ep01_intro"])

            updated = json.loads(story_state_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["last_completed_episode"], "ep01_intro")
            self.assertEqual(updated["episode_summaries"], {"ep01_intro": {"cycle_count": 2}})
            self.assertEqual(updated["character_states"], {})
            self.assertEqual(updated["active_clues"], {})
            self.assertEqual(updated["arc_position"], {})
            self.assertTrue((backup_dir / "story_state_before_reset.json").exists())
            self.assertTrue((backup_dir / "story_state_removed_entries.json").exists())

    def test_reset_story_state_from_episode_removes_target_and_later_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            story_state_path = Path(tmpdir) / "story_state.json"
            backup_dir = Path(tmpdir) / "backup"
            story_state_path.write_text(
                json.dumps(
                    {
                        "last_completed_episode": "ep02_followup",
                        "arc_position": {"act": 2},
                        "character_states": {"kim_sumin": {"mood": "alert"}},
                        "active_clues": {"cipher": {"status": "open"}},
                        "episode_summaries": {
                            "ep01_intro": {"cycle_count": 2},
                            "ep02_followup": {"cycle_count": 1},
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = daily_pipeline._reset_story_state_from_episode(
                story_state_path,
                "ep01_intro",
                backup_dir,
            )

            self.assertTrue(result["changed"])
            self.assertEqual(result["removed_episode_keys"], ["ep01_intro", "ep02_followup"])
            self.assertEqual(result["remaining_episode_keys"], [])

            updated = json.loads(story_state_path.read_text(encoding="utf-8"))
            self.assertIsNone(updated["last_completed_episode"])
            self.assertEqual(updated["episode_summaries"], {})
            self.assertEqual(updated["character_states"], {})
            self.assertEqual(updated["active_clues"], {})
            self.assertEqual(updated["arc_position"], {})


if __name__ == "__main__":
    unittest.main()

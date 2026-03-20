#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.novel_writer import database


class DatabaseEpisodeRunSelectionTest(unittest.TestCase):
    def test_load_episode_interactions_prefers_latest_simulate_run(self) -> None:
        old_db_path = database.DB_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            database.DB_PATH = str(Path(tmpdir) / "simulation.db")
            try:
                database.set_tracking_context(reset=True)
                database.init_db()

                run1 = database.begin_episode_run("ep_test", source="simulate")
                database.save_interaction(
                    SimpleNamespace(
                        id="ix-old",
                        episode_id="ep_test",
                        turn=1,
                        speaker_id="kim_sumin",
                        speaker_name="수민",
                        content="첫 번째 런의 대사",
                        action_type="dialogue",
                        target_id=None,
                        timestamp=datetime.utcnow(),
                        metadata={},
                    )
                )
                database.finish_episode_run(run1)

                database.set_tracking_context(reset=True)
                run2 = database.begin_episode_run("ep_test", source="simulate")
                database.save_interaction(
                    SimpleNamespace(
                        id="ix-new",
                        episode_id="ep_test",
                        turn=1,
                        speaker_id="kim_sumin",
                        speaker_name="수민",
                        content="두 번째 런의 최신 대사",
                        action_type="dialogue",
                        target_id=None,
                        timestamp=datetime.utcnow(),
                        metadata={},
                    )
                )
                database.finish_episode_run(run2)

                database.set_tracking_context(reset=True)
                latest = database.load_episode_interactions("ep_test")
                all_rows = database.load_episode_interactions("ep_test", latest_only=False)

                self.assertEqual(len(latest), 1)
                self.assertEqual(latest[0]["id"], "ix-new")
                self.assertEqual(len(all_rows), 2)
            finally:
                database.set_tracking_context(reset=True)
                database.DB_PATH = old_db_path

    def test_archive_and_purge_episodes_from_removes_target_and_later(self) -> None:
        old_db_path = database.DB_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            database.DB_PATH = str(Path(tmpdir) / "simulation.db")
            try:
                database.set_tracking_context(reset=True)
                database.init_db()

                for episode_id in ("ep01_alpha", "ep02_beta", "ep03_gamma"):
                    database.upsert_episode(episode_id, {"beats": []})
                    run_id = database.begin_episode_run(episode_id, source="simulate")
                    database.save_interaction(
                        SimpleNamespace(
                            id=f"ix-{episode_id}",
                            episode_id=episode_id,
                            turn=1,
                            speaker_id="kim_sumin",
                            speaker_name="수민",
                            content=f"{episode_id} 대사",
                            action_type="dialogue",
                            target_id=None,
                            timestamp=datetime.utcnow(),
                            metadata={},
                        )
                    )
                    database.save_emotion(
                        "kim_sumin",
                        episode_id,
                        1,
                        "focus",
                        0.7,
                    )
                    database.finish_episode_run(run_id)

                backup_dir = Path(tmpdir) / "backup"
                result = database.archive_and_purge_episodes_from("ep02_beta", backup_dir)

                self.assertEqual(result["episode_number"], 2)
                self.assertEqual(result["episode_ids"], ["ep02_beta", "ep03_gamma"])

                archive_path = backup_dir / "database_episode_archive.json"
                self.assertTrue(archive_path.exists())
                archive = json.loads(archive_path.read_text(encoding="utf-8"))
                self.assertEqual(archive["episode_ids"], ["ep02_beta", "ep03_gamma"])
                self.assertIn("interactions", archive["tables"])
                self.assertIn("episode_runs", archive["tables"])
                self.assertIn("episodes", archive["tables"])

                conn = database._connect()
                try:
                    live_interactions = conn.execute(
                        "SELECT DISTINCT episode_id FROM interactions ORDER BY episode_id"
                    ).fetchall()
                    live_episodes = conn.execute(
                        "SELECT id FROM episodes ORDER BY id"
                    ).fetchall()
                finally:
                    conn.close()

                self.assertEqual([row["episode_id"] for row in live_interactions], ["ep01_alpha"])
                self.assertEqual([row["id"] for row in live_episodes], ["ep01_alpha"])
            finally:
                database.set_tracking_context(reset=True)
                database.DB_PATH = old_db_path


if __name__ == "__main__":
    unittest.main()

# Release Notes (2026-03-07)

- Deployment commit: `181a16d` (`a112304..181a16d`)
- Change size: **65 files changed**, `+3368 / -851`

## Highlights

1. Story and configuration expansion
- Updated characters, world facts, storyline, and multiple episode configs.
- Target episodes: `ep12, ep17, ep28-29, ep31-37, ep39-41`

2. Generation pipeline improvements
- Improved chapter/simulation flow and quality-related logic.
- Key files: `generate_chapter.py`, `simulate.py`, `trial_simulate.py`

3. Core engine refinement
- Enhanced orchestration, directing, prose generation, scene distillation, RL policy handling, and data/config layers.
- Key files:
  - `src/novel_writer/orchestrator.py`
  - `src/novel_writer/director.py`
  - `src/novel_writer/prose_generator.py`
  - `src/novel_writer/scene_distiller.py`
  - `src/novel_writer/rl_policy.py`
  - `src/novel_writer/database.py`
  - `src/novel_writer/config_loader.py`
  - `src/novel_writer/models.py`

4. New utilities and tooling
- Added environment loader: `src/novel_writer/env_loader.py`
- Added benchmark/review/automation tools:
  - `tools/build_ep_benchmark_notebook.py`
  - `tools/llm_chapter_review_benchmark.py`
  - `tools/eval_all_episodes_cycle.sh`
  - `tools/run_ep01_ep10_benchmark_with_good_and_llm.sh`
  - `tools/archive_workspace_state.sh`

5. Experiment and analysis assets
- Updated notebooks:
  - `notebooks/benchmark_results_explorer.ipynb`
  - `notebooks/emotion_persona_relationship_explorer.ipynb`
  - `notebooks/simulation_iteration_tracking_explorer.ipynb`
- Updated RL policy data: `data/rl_policy.json`

6. Cleanup and snapshots
- Removed: `tools/quality_adaptive_generator.py`
- Added backup snapshots under: `backup/run_reset_*`

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

# Release Notes (2026-03-08)

- Branch: `main`
- Scope: Reader-feedback-driven prose quality improvements + loop automation updates

## Highlights

1. Prose readability and pacing upgrades
- Expanded `src/novel_writer/prose_generator.py` with stronger readability controls.
- Added rhythm constraints, technical-term gloss guidance, and safer paragraph normalization.
- Reinforced transition/polish behavior to reduce dense analytical flow and improve breathing points.

2. Review-feedback integration path
- Added `src/novel_writer/review_feedback.py` for handling review-derived improvement signals.
- Updated orchestration and simulation flow to incorporate review-informed adjustments:
  - `src/novel_writer/orchestrator.py`
  - `simulate.py`
  - `trial_simulate.py`
  - `src/novel_writer/trial_runner.py`

3. Discord loop bot and fixer agent improvements
- Updated `tools/discord_loop_bot.py` to improve multi-agent loop behavior.
- Converted fixer-agent prompt instructions to Korean while preserving strict JSON schema compatibility.
- Updated loop state tracking in `data/discord_loop_state.json`.

4. Story configuration refresh
- Updated episode and storyline configs:
  - `config/episodes/ep36_patron_failsafe.yaml`
  - `config/episodes/ep38_showdown_quantum_part2.yaml`
  - `config/episodes/ep40_fbi_cutoff.yaml`
  - `config/episodes/ep41_epilogue.yaml`
  - `config/storyline.yaml`

5. Generation entrypoint updates
- Updated `generate_chapter.py` and related generation wiring for the latest loop/review flow.

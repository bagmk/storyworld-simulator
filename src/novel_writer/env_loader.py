from __future__ import annotations

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


def load_project_env(start_dir: str | Path | None = None) -> Path | None:
    """
    Load a local `.env` file into process env without overwriting existing vars.

    Supports simple `KEY=VALUE` lines and quoted values.
    """
    start = Path(start_dir or Path.cwd()).resolve()
    env_path = _find_env_file(start)
    if env_path is None:
        return None

    loaded = 0
    with env_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue

            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
            loaded += 1

    if loaded:
        logger.debug("Loaded %d env var(s) from %s", loaded, env_path)
    return env_path


def _find_env_file(start: Path) -> Path | None:
    for candidate_dir in [start, *start.parents]:
        candidate = candidate_dir / ".env"
        if candidate.is_file():
            return candidate
    return None

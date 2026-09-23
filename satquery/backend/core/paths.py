"""
SatQuery AI — Central Project Path Manager
============================================

Single source of truth for repository paths.

Never build paths using:
    ../../models
    ../../../configs

from individual modules.

Everything should resolve from PROJECT_ROOT.
"""

from __future__ import annotations

from pathlib import Path


# paths.py
# satquery/backend/core/paths.py

# backend/core -> backend -> satquery -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
DATA_DIR = PROJECT_ROOT / "satquery" / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"


def repo_path(*parts: str | Path) -> Path:
    """
    Resolve a path relative to the repository root.
    """
    return PROJECT_ROOT.joinpath(*parts)


def config_path(filename: str) -> Path:
    return CONFIG_DIR / filename


def model_path(filename: str) -> Path:
    return MODELS_DIR / filename


def checkpoint_path(filename: str) -> Path:
    return CHECKPOINT_DIR / filename


def ensure_runtime_dirs() -> None:
    """
    Create runtime directories if they do not exist.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def path_status(path: Path) -> dict:
    """
    Return diagnostic information about a path.
    """
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def get_project_status() -> dict:
    """
    Diagnostic information used by preflight/health endpoints.
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "config_dir": str(CONFIG_DIR),
        "models_dir": str(MODELS_DIR),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "paths_exist": {
            "config": CONFIG_DIR.exists(),
            "models": MODELS_DIR.exists(),
            "checkpoints": CHECKPOINT_DIR.exists(),
            "data": DATA_DIR.exists(),
            "results": RESULTS_DIR.exists(),
        },
    }

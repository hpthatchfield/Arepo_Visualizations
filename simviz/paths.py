"""Portable paths relative to the repository root."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SNAPS_DIR = REPO_ROOT / "sample_snaps"
EXAMPLE_OUTPUT_DIR = REPO_ROOT / "example_output"
EXAMPLES_DIR = REPO_ROOT / "examples"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Optional CMZ simulation data kept outside the repo (override with SIMVIZ_CMZ_DATA_DIR).
_DEFAULT_CMZ = Path.home() / "Research" / "Archive" / "Old_Code" / "arepo_CMZ" / "TS_2020"
CMZ_DATA_DIR = Path(os.environ.get("SIMVIZ_CMZ_DATA_DIR", _DEFAULT_CMZ))

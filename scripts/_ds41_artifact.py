#!/usr/bin/env python3
"""Shared model path for DS41 scripts; source of truth is runtime/ds41/artifact.json."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path('/home/funboy/StrixHaloClusterDS41')
ARTIFACT = json.loads((REPO / 'runtime/ds41/artifact.json').read_text())
MODEL_DIR = Path(ARTIFACT['model_dir'])
MODEL_FILE = MODEL_DIR / ARTIFACT['model_file']
ORIGINAL_MODEL_DIR = Path(ARTIFACT['original_model_dir'])

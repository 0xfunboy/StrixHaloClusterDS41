#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
spec=importlib.util.spec_from_file_location('base_sidecar',ROOT/'scripts/serve-ds4-usable-tokenizer.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.MAX_MODEL_LEN=65536
base.ThreadingHTTPServer(('127.0.0.1',19223),base.H).serve_forever()

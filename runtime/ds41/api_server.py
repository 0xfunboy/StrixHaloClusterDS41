"""DS41 vLLM API entrypoint: register the pinned GGUF plugin before CLI parsing."""
from __future__ import annotations

import os

from vllm_gguf_plugin import register

register()

if os.environ.get("DS41_NATIVE_HIP_MOE", "0") == "1":
    from runtime.ds41.native_hip_moe_runtime import ensure_loaded

    ensure_loaded()

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    main()

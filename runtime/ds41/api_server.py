"""DS41 vLLM API entrypoint: register the pinned GGUF plugin before CLI parsing."""
from vllm_gguf_plugin import register

register()

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main
    main()

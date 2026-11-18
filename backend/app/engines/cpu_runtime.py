"""CPU constraints shared by local inference subprocesses (no framework imports)."""


def cpu_llama_args(threads: int) -> list[str]:
    # Decoder, vision projector, KV cache AND auxiliary tensor operations must
    # remain on the CPU. Setting only gpu-layers=0 does not guarantee that.
    return [
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "--no-mmproj-offload",
        "--no-kv-offload",
        "--no-op-offload",
        "--flash-attn",
        "off",
        "--fit",
        "off",
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads),
    ]


def cpu_subprocess_env(env: dict[str, str], threads: int) -> dict[str, str]:
    result = {k: v for k, v in env.items() if not k.startswith("LLAMA_ARG_")}
    result.pop("MTMD_BACKEND_DEVICE", None)
    result.update(
        CUDA_VISIBLE_DEVICES="-1",
        HIP_VISIBLE_DEVICES="-1",
        ROCR_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        BLIS_NUM_THREADS=str(threads),
        ARENA_CPU_ONLY="1",
        TORCH_DEVICE="cpu",
    )
    return result

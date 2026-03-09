#!/usr/bin/env python3
"""
Benchmark FP4 vs BF16 matmul throughput with pre-quantized weights.
Activation quantization is included in timed regions, weight quantization is not.
"""

import argparse
import os
import re
import shutil
import subprocess
import time
import torch

# Use a dedicated workspace to avoid stale build.ninja settings from previous runs.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer_fp4_bench")

def _compiler_major_version(compiler: str):
    try:
        output = subprocess.check_output(
            [compiler, "-dumpfullversion", "-dumpversion"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    m = re.match(r"(\d+)", output)
    return int(m.group(1)) if m else None


def _configure_flashinfer_host_compiler():
    """
    FlashInfer JIT forwards env `CC` to nvcc via `-ccbin`.
    CUDA 12.x rejects GCC > 14 unless `-allow-unsupported-compiler` is set.
    """
    cc = os.environ.get("CC", "")
    cc_bin = cc.split()[0] if cc else ""
    cc_major = _compiler_major_version(cc_bin) if cc_bin else None

    for gcc_candidate, gxx_candidate in (
        ("gcc-14", "g++-14"),
        ("gcc-13", "g++-13"),
    ):
        gcc_path = shutil.which(gcc_candidate)
        gxx_path = shutil.which(gxx_candidate)
        if gcc_path and gxx_path:
            major = _compiler_major_version(gcc_path)
            if major is not None and major <= 14:
                should_override = (
                    not cc_bin
                    or cc_major is None
                    or cc_major > 14
                    or "conda-linux-gnu-cc" in cc_bin
                )
                if should_override:
                    os.environ["CC"] = gcc_path
                    os.environ["CXX"] = gxx_path
                    prior = f"{cc_bin} (gcc {cc_major})" if cc_bin else "<unset>"
                    print(f"[FlashInfer] Setting CC: {prior} -> {gcc_path}")
                return

    extra_cudaflags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "").strip()
    fallback_flag = "-allow-unsupported-compiler"
    if fallback_flag not in extra_cudaflags.split():
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (
            f"{extra_cudaflags} {fallback_flag}".strip()
        )
        print(
            "[FlashInfer] No GCC<=14 found; using nvcc fallback flag "
            "'-allow-unsupported-compiler'."
        )


_configure_flashinfer_host_compiler()

from flashinfer import nvfp4_quantize, mm_fp4, SfLayout

x_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)

BASE_VIDEO_SEQUENCE_LENGTH = 8160
BASE_AUDIO_SEQUENCE_LENGTH = 126
SEQUENCE_LENGTH_PRESETS = [
    {
        "base_video_sequence_length": 8160,
        "refinement_video_sequence_length": 32640,
        "audio_sequence_length": 126,
    },
    {
        "base_video_sequence_length": 15810,
        "refinement_video_sequence_length": 63240,
        "audio_sequence_length": 251,
    },
    {
        "base_video_sequence_length": 23460,
        "refinement_video_sequence_length": 93840,
        "audio_sequence_length": 376,
    },
]

LTX2_LOGGED_GEMM_SHAPES: list[tuple[str, int, int, int]] = [
    ("ltx2.adaln_single.linear", 8160, 36864, 4096),
    ("ltx2.prompt_adaln_single.linear", 1, 8192, 4096),
    ("ltx2.av_ca_video_scale_shift_adaln_single.linear", 1, 16384, 4096),
    ("ltx2.av_ca_a2v_gate_adaln_single.linear", 1, 4096, 4096),
    ("ltx2.audio_adaln_single.linear", 126, 18432, 2048),
    ("ltx2.audio_prompt_adaln_single.linear", 1, 4096, 2048),
    ("ltx2.av_ca_audio_scale_shift_adaln_single.linear", 1, 8192, 2048),
    ("ltx2.av_ca_v2a_gate_adaln_single.linear", 1, 2048, 2048),
    ("ltx2.blocks.0.attn1.to_gate_logits", 8160, 32, 4096),
    ("ltx2.blocks.0.attn1.to_q", 8160, 4096, 4096),
    ("ltx2.blocks.0.attn1.to_k", 8160, 4096, 4096),
    ("ltx2.blocks.0.attn1.to_v", 8160, 4096, 4096),
    ("ltx2.blocks.0.attn1.to_out", 8160, 4096, 4096),
    ("ltx2.blocks.0.attn2.to_q", 8160, 4096, 4096),
    ("ltx2.blocks.0.attn2.to_k", 1024, 4096, 4096),
    ("ltx2.blocks.0.attn2.to_v", 1024, 4096, 4096),
    ("ltx2.blocks.0.attn2.to_gate_logits", 8160, 32, 4096),
    ("ltx2.blocks.0.attn2.to_out", 8160, 4096, 4096),
    ("ltx2.blocks.0.audio_attn1.to_q", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_attn1.to_k", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_attn1.to_v", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_attn1.to_gate_logits", 126, 32, 2048),
    ("ltx2.blocks.0.audio_attn1.to_out", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_attn2.to_q", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_attn2.to_k", 1024, 2048, 2048),
    ("ltx2.blocks.0.audio_attn2.to_v", 1024, 2048, 2048),
    ("ltx2.blocks.0.audio_attn2.to_gate_logits", 126, 32, 2048),
    ("ltx2.blocks.0.audio_attn2.to_out", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_to_video_attn.to_q", 8160, 2048, 4096),
    ("ltx2.blocks.0.audio_to_video_attn.to_k", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_to_video_attn.to_v", 126, 2048, 2048),
    ("ltx2.blocks.0.audio_to_video_attn.to_gate_logits", 8160, 32, 4096),
    ("ltx2.blocks.0.audio_to_video_attn.to_out", 8160, 4096, 2048),
    ("ltx2.blocks.0.video_to_audio_attn.to_q", 126, 2048, 2048),
    ("ltx2.blocks.0.video_to_audio_attn.to_k", 8160, 2048, 4096),
    ("ltx2.blocks.0.video_to_audio_attn.to_v", 8160, 2048, 4096),
    ("ltx2.blocks.0.video_to_audio_attn.to_gate_logits", 126, 32, 2048),
    ("ltx2.blocks.0.video_to_audio_attn.to_out", 126, 2048, 2048),
    ("ltx2.blocks.0.ffn.fc_in", 8160, 16384, 4096),
    ("ltx2.blocks.0.ffn.fc_out", 8160, 4096, 16384),
    ("ltx2.blocks.0.audio.ffn.fc_in", 126, 8192, 2048),
    ("ltx2.blocks.0.audio.ffn.fc_out", 126, 2048, 8192),
]


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


# ThunderKittens benchmarking code is intentionally commented out for now.


def benchmark_bf16_matmul_tflops(x_bf16, weight_bf16, num_runs=50, warmup_runs=5):
    """Benchmark BF16 matmul only."""
    device = x_bf16.device
    bsz, seq_len, in_features = x_bf16.shape
    out_features = weight_bf16.shape[0]

    x_2d = x_bf16.view(-1, in_features)
    w_t = weight_bf16.t()

    theoretical_flops = 2 * bsz * seq_len * in_features * out_features

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = torch.matmul(x_2d, w_t)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            out = torch.matmul(x_2d, w_t)
            out = out + 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start
    return {
        "tflops_per_second": (theoretical_flops * num_runs) / total_time / 1e12,
        "avg_time_per_run_ms": total_time / num_runs * 1000,
    }


def benchmark_fp4_prequant_weight_tflops(x_bf16, weight_bf16, num_runs=50, warmup_runs=5):
    """Benchmark FP4 with pre-quantized weights and timed activation quantization."""
    device = x_bf16.device
    bsz, seq_len, in_features = x_bf16.shape
    out_features = weight_bf16.shape[0]

    x_2d = x_bf16.view(-1, in_features)

    # Pre-quantize weights once so weight quantization overhead is excluded from timing.
    w_global_sf = (448 * 6) / weight_bf16.float().abs().nan_to_num().max()
    w_fp4, w_scale = nvfp4_quantize(
        weight_bf16,
        w_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )

    theoretical_flops = 2 * bsz * seq_len * in_features * out_features

    with torch.no_grad():
        for _ in range(warmup_runs):
            # x_global_sf = (448 * 6) / x_2d.float().abs().nan_to_num().max()
            x_fp4, x_scale = nvfp4_quantize(
                x_2d,
                x_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            out_scale = 1.0 / (x_global_sf * w_global_sf)
            _ = mm_fp4(
                x_fp4,
                w_fp4.t(),
                x_scale,
                w_scale.t(),
                out_scale,
                torch.bfloat16,
                None,
                backend="cutlass",
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            # x_global_sf = (448 * 6) / x_2d.float().abs().nan_to_num().max()
            x_fp4, x_scale = nvfp4_quantize(
                x_2d,
                x_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            out_scale = 1.0 / (x_global_sf * w_global_sf)
            out = mm_fp4(
                x_fp4,
                w_fp4.t(),
                x_scale,
                w_scale.t(),
                out_scale,
                torch.bfloat16,
                None,
                backend="cutlass",
            )
            out = out + 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start
    return {
        "tflops_per_second": (theoretical_flops * num_runs) / total_time / 1e12,
        "avg_time_per_run_ms": total_time / num_runs * 1000,
    }


# ThunderKittens benchmark and precision helpers are intentionally commented out for now.


def build_configs_for_video_sequence_length(
    video_sequence_length: int,
    audio_sequence_length: int,
):
    configs = []
    for name, m, n, k in LTX2_LOGGED_GEMM_SHAPES:
        if m == BASE_VIDEO_SEQUENCE_LENGTH:
            seq_len = video_sequence_length
        elif m == BASE_AUDIO_SEQUENCE_LENGTH:
            seq_len = audio_sequence_length
        else:
            seq_len = m
        configs.append(
            {
                "batch_size": 1,
                "seq_len": seq_len,
                "in_features": k,
                "out_features": n,
                "name": name,
            }
        )
    return configs


def run_benchmark(
    num_runs=50,
    warmup_runs=5,
):
    print("=== FP4 vs BF16 Benchmark (Weight Pre-Quantized, Activation Quant Timed) ===")

    if not torch.cuda.is_available():
        print("CUDA is required for Flashinfer FP4 benchmark, but no CUDA device is available.")
        return

    device = "cuda"
    print(f"Using device: {device}")
    print("TK NVFP4 benchmarking: commented out")

    def run_section(configs, section_name):
        fp4_faster_layers = []

        print(f"\n=== {section_name} ===")
        print("matmul-performance (weight quant excluded for FP4):")
        print(
            f"{'idx':>4} {'matmul':<48}{'M':>12}{'N':>12}{'K':>12}"
            f"{'Torch-BF16':>14}{'Flashinfer-fp4':>16}"
        )

        for i, config in enumerate(configs):
            bsz = config["batch_size"]
            seq_len = config["seq_len"]
            in_features = config["in_features"]
            out_features = config["out_features"]

            torch.manual_seed(42)
            x = torch.randn(bsz, seq_len, in_features, dtype=torch.bfloat16, device=device)
            w = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)

            bf16_results = benchmark_bf16_matmul_tflops(
                x,
                w,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
            )
            fp4_results = benchmark_fp4_prequant_weight_tflops(
                x,
                w,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
            )

            print(
                f"{i:>4} {config['name']:<48}"
                f"{seq_len:>12.1f}"
                f"{out_features:>12.1f}{in_features:>12.1f}"
                f"{bf16_results['tflops_per_second']:>14.6f}{fp4_results['tflops_per_second']:>16.6f}"
            )

            if fp4_results["tflops_per_second"] > bf16_results["tflops_per_second"]:
                speedup = fp4_results["tflops_per_second"] / bf16_results["tflops_per_second"]
                fp4_faster_layers.append(
                    {
                        "name": config["name"],
                        "bf16_tflops": bf16_results["tflops_per_second"],
                        "fp4_tflops": fp4_results["tflops_per_second"],
                        "speedup": speedup,
                    }
                )
            # Release large per-shape tensors before moving to the next shape.
            del x, w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nLayers where Flashinfer-fp4 has higher benchmarked TFLOPs (weight quant excluded):")
        if fp4_faster_layers:
            for layer in fp4_faster_layers:
                print(
                    f"  - {layer['name']}: "
                    f"{layer['speedup']:.4f}x "
                    f"(BF16={layer['bf16_tflops']:.6f}, FP4={layer['fp4_tflops']:.6f})"
                )
        else:
            print("  (none)")

    for preset in SEQUENCE_LENGTH_PRESETS:
        base_video_sequence_length = preset["base_video_sequence_length"]
        refinement_video_sequence_length = preset["refinement_video_sequence_length"]
        audio_sequence_length = preset["audio_sequence_length"]
        run_section(
            build_configs_for_video_sequence_length(
                base_video_sequence_length,
                audio_sequence_length=audio_sequence_length,
            ),
            f"Base Stage | video_seq_len={base_video_sequence_length} | audio_seq_len={audio_sequence_length}",
        )
        run_section(
            build_configs_for_video_sequence_length(
                refinement_video_sequence_length,
                audio_sequence_length=audio_sequence_length,
            ),
            f"Refinement Stage | video_seq_len={refinement_video_sequence_length} | audio_seq_len={audio_sequence_length}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=50, help="Timed iterations per shape")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warmup iterations per shape")
    args = parser.parse_args()

    run_benchmark(
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
    )

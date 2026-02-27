#!/usr/bin/env python3
"""
Benchmark FP4 vs BF16 matmul throughput with pre-quantized weights.
Activation quantization is included in timed regions, weight quantization is not.
"""

import argparse
import os
import time
import torch

# Redirect FlashInfer workspace/cache to a writable path in constrained envs.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")

from flashinfer import nvfp4_quantize, mm_fp4, SfLayout

x_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)


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


def run_benchmark(num_runs=50, warmup_runs=5):
    print("=== FP4 vs BF16 Benchmark (Weight Pre-Quantized, Activation Quant Timed) ===")

    if not torch.cuda.is_available():
        print("CUDA is required for Flashinfer FP4 benchmark, but no CUDA device is available.")
        return

    device = "cuda"
    print(f"Using device: {device}")

    base_configs = [
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_q"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_k"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_v"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_out"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_q"},
        {"batch_size": 1, "seq_len": 1024, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_k"},
        {"batch_size": 1, "seq_len": 1024, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_v"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_out"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_q"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_k"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_v"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_out"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_q"},
        {"batch_size": 1, "seq_len": 1024, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_k"},
        {"batch_size": 1, "seq_len": 1024, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_v"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_out"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 2048, "name": "ltx2.blocks.32.audio_to_video_attn.to_q"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_to_video_attn.to_k"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_to_video_attn.to_v"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 2048, "out_features": 4096, "name": "ltx2.blocks.32.audio_to_video_attn.to_out"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_q"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_k"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_v"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_out"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 4096, "out_features": 16384, "name": "ltx2.blocks.32.ffn.fc_in"},
        {"batch_size": 1, "seq_len": 32640, "in_features": 16384, "out_features": 4096, "name": "ltx2.blocks.32.ffn.fc_out"},
        {"batch_size": 1, "seq_len": 126, "in_features": 2048, "out_features": 8192, "name": "ltx2.blocks.32.audio.ffn.fc_in"},
        {"batch_size": 1, "seq_len": 126, "in_features": 8192, "out_features": 2048, "name": "ltx2.blocks.32.audio.ffn.fc_out"},

        {'batch_size': 1, 'seq_len': 32640, 'in_features': 4096, 'out_features': 3*4096, 'name': 'ltx2.blocks.32.attn1.fused_proj'},
    ]

    def build_scaled_configs(configs, divisor):
        scaled = []
        for cfg in configs:
            cfg_scaled = dict(cfg)
            cfg_scaled["seq_len"] = max(1, cfg["seq_len"] // divisor)
            cfg_scaled["name"] = f"{cfg['name']} [seq/{divisor}]"
            scaled.append(cfg_scaled)
        return scaled

    def run_section(configs, section_name):
        fp4_faster_layers = []

        print(f"\n=== {section_name} ===")
        print("matmul-performance (weight quant excluded for FP4):")
        print(f"{'idx':>4} {'matmul':<48}{'N':>12}{'K':>12}{'Torch-BF16':>14}{'Flashinfer-fp4':>16}")

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
                f"{in_features:>12.1f}{out_features:>12.1f}"
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

    run_section(base_configs, "Original Sequence Length")
    run_section(build_scaled_configs(base_configs, 4), "Sequence Length Divided By 4")
    run_section(build_scaled_configs(base_configs, 8), "Sequence Length Divided By 8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=50, help="Timed iterations per shape")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warmup iterations per shape")
    args = parser.parse_args()

    run_benchmark(num_runs=args.num_runs, warmup_runs=args.warmup_runs)

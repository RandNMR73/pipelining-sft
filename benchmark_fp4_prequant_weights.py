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
import sys
import time
from pathlib import Path
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


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _try_import_tk_nvfp4_ops():
    """
    Returns:
        tuple[dict[str, callable], nvfp4_quantize, nvfp4_quantize_with_global|None]
        or None if TK extension is unavailable.
    """
    tk_dir = Path(__file__).resolve().parent / "ThunderKittens" / "kernels" / "gemm" / "nvfp4_b200"
    if not tk_dir.exists():
        print(f"[TK] Skipping TK NVFP4 benchmark: path not found: {tk_dir}")
        return None

    ext_suffix = None
    try:
        import sysconfig
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    except Exception:
        pass
    expected_ext_name = f"_C{ext_suffix}" if ext_suffix else "_C<EXT_SUFFIX>"

    if "_C" in sys.modules:
        mod = sys.modules["_C"]
        mod_file = getattr(mod, "__file__", None)
        if mod_file and str(tk_dir) not in str(mod_file):
            print(f"[TK] Skipping TK NVFP4 benchmark: '_C' already loaded from {mod_file}")
            return None

    sys.path.insert(0, str(tk_dir))
    try:
        import _C as tk_mod  # type: ignore
        tk_nvfp4_quantize = tk_mod.nvfp4_quantize
        tk_nvfp4_quantize_with_global = getattr(tk_mod, "nvfp4_quantize_with_global", None)
        gemm_variants = {"cfg_default": tk_mod.nvfp4_gemm}
        cfg_entries = []
        for attr in dir(tk_mod):
            m = re.fullmatch(r"nvfp4_gemm_cfg(\d+)", attr)
            if m is None:
                continue
            cfg_entries.append((int(m.group(1)), attr))
        for cfg_id, fn_name in sorted(cfg_entries):
            gemm_variants[f"cfg{cfg_id}"] = getattr(tk_mod, fn_name)
        return gemm_variants, tk_nvfp4_quantize, tk_nvfp4_quantize_with_global
    except Exception as e:
        print(f"[TK] Skipping TK NVFP4 benchmark: failed to import extension from {tk_dir}: {e}")
        print(
            f"[TK] Build hint: activate the same Python env used for this benchmark ({sys.executable}), "
            f"then run:\n"
            f"      cd {tk_dir}\n"
            f"      make clean\n"
            f"      make CONFIG=pytorch OUT=\"{expected_ext_name}\""
        )
        return None
    finally:
        if sys.path and sys.path[0] == str(tk_dir):
            sys.path.pop(0)


class TKFP4LinearPrequant:
    """
    Thin TK FP4 linear wrapper for benchmarking.
    Uses two-level quantization API from ThunderKittens test_gemm.py:
      - nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, scale_2d)
      - nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    """

    def __init__(
        self,
        weight_bf16,
        tk_nvfp4_gemm_fn,
        tk_nvfp4_quantize,
        tk_nvfp4_quantize_with_global=None,
        scale_2d=False,
    ):
        self.tk_nvfp4_gemm = tk_nvfp4_gemm_fn
        self.tk_nvfp4_quantize = tk_nvfp4_quantize
        self.tk_nvfp4_quantize_with_global = tk_nvfp4_quantize_with_global
        self.scale_2d = scale_2d

        self.out_features = weight_bf16.shape[0]
        self.in_features = weight_bf16.shape[1]
        self.n_pad = _round_up(self.out_features, 128)
        self.k_pad = _round_up(self.in_features, 64)

        weight_padded = torch.zeros(
            self.n_pad,
            self.k_pad,
            device=weight_bf16.device,
            dtype=torch.bfloat16,
        )
        weight_padded[: self.out_features, : self.in_features] = weight_bf16
        self.weight_padded = weight_padded.contiguous()

        self.B_fp4x2 = torch.empty(
            self.n_pad,
            self.k_pad // 2,
            dtype=torch.float4_e2m1fn_x2,
            device=weight_bf16.device,
        )
        self.B_sc = torch.empty(
            self.n_pad // 128,
            self.k_pad // 64,
            512,
            dtype=torch.float8_e4m3fn,
            device=weight_bf16.device,
        )
        self.B_sc_global = torch.empty(1, dtype=torch.float32, device=weight_bf16.device)
        if self.tk_nvfp4_quantize_with_global is not None:
            # Match FlashInfer's weight global scale convention:
            # gsf = (448*6)/amax  => dequant global = 1/gsf = amax/(448*6)
            w_amax = self.weight_padded.float().abs().nan_to_num().max()
            self.B_sc_global.fill_(w_amax / (448.0 * 6.0))
            self.tk_nvfp4_quantize_with_global(
                self.weight_padded, self.B_fp4x2, self.B_sc, self.B_sc_global, self.scale_2d
            )
        else:
            self.tk_nvfp4_quantize(
                self.weight_padded, self.B_fp4x2, self.B_sc, self.B_sc_global, self.scale_2d
            )

        self._cached_m_pad = None
        self._cached_direct = None
        self._A_bf16 = None
        self._A_fp4x2 = None
        self._A_sc = None
        self._A_sc_global = None
        self._C = None

    def _ensure_activation_buffers(self, m, direct):
        # TK GEMM kernel uses Mb=256 tile height; pad M to 256 so at least one row block runs.
        m_pad = _round_up(m, 256)
        if self._cached_m_pad == m_pad and self._cached_direct == direct:
            return

        device = self.weight_padded.device
        if direct:
            self._A_bf16 = None
            self._A_fp4x2 = torch.empty(m, self.k_pad // 2, dtype=torch.float4_e2m1fn_x2, device=device)
            self._A_sc = torch.empty(m // 128, self.k_pad // 64, 512, dtype=torch.float8_e4m3fn, device=device)
            self._C = torch.zeros(m, self.n_pad, dtype=torch.bfloat16, device=device)
        else:
            self._A_bf16 = torch.zeros(m_pad, self.k_pad, dtype=torch.bfloat16, device=device)
            self._A_fp4x2 = torch.empty(m_pad, self.k_pad // 2, dtype=torch.float4_e2m1fn_x2, device=device)
            self._A_sc = torch.empty(m_pad // 128, self.k_pad // 64, 512, dtype=torch.float8_e4m3fn, device=device)
            self._C = torch.zeros(m_pad, self.n_pad, dtype=torch.bfloat16, device=device)
        self._A_sc_global = torch.empty(1, dtype=torch.float32, device=device)
        self._cached_m_pad = m_pad
        self._cached_direct = direct

    def __call__(self, x_2d):
        m, k = x_2d.shape
        assert k == self.in_features, f"Expected K={self.in_features}, got {k}"
        # Direct path only when M already matches TK GEMM tile height.
        direct = (m % 256 == 0) and (k % 64 == 0)
        self._ensure_activation_buffers(m, direct)

        if direct:
            if self.tk_nvfp4_quantize_with_global is not None:
                # Activation global scale fixed at 1 (matches requested setup).
                self._A_sc_global.fill_(1.0)
                self.tk_nvfp4_quantize_with_global(
                    x_2d, self._A_fp4x2, self._A_sc, self._A_sc_global, self.scale_2d
                )
            else:
                self.tk_nvfp4_quantize(x_2d, self._A_fp4x2, self._A_sc, self._A_sc_global, self.scale_2d)
        else:
            self._A_bf16.zero_()
            self._A_bf16[:m, :k] = x_2d
            if self.tk_nvfp4_quantize_with_global is not None:
                self._A_sc_global.fill_(1.0)
                self.tk_nvfp4_quantize_with_global(
                    self._A_bf16, self._A_fp4x2, self._A_sc, self._A_sc_global, self.scale_2d
                )
            else:
                self.tk_nvfp4_quantize(
                    self._A_bf16, self._A_fp4x2, self._A_sc, self._A_sc_global, self.scale_2d
                )
        self.tk_nvfp4_gemm(
            self._A_fp4x2,
            self._A_sc,
            self._A_sc_global,
            self.B_fp4x2,
            self.B_sc,
            self.B_sc_global,
            self._C,
        )
        return self._C[:m, : self.out_features]


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


def benchmark_tk_fp4_prequant_weight_tflops(
    x_bf16,
    weight_bf16,
    tk_gemm_variants,
    tk_nvfp4_quantize,
    tk_nvfp4_quantize_with_global=None,
    num_runs=50,
    warmup_runs=5,
    scale_2d=False,
    autotune_runs=5,
    min_cosine_vs_bf16=0.95,
    cosine_check_max_rows=2048,
):
    """Benchmark TK NVFP4 GEMM with pre-quantized weights and timed activation quantization."""
    device = x_bf16.device
    bsz, seq_len, in_features = x_bf16.shape
    out_features = weight_bf16.shape[0]
    x_2d = x_bf16.view(-1, in_features)
    # Use a bounded subset for cosine gating to avoid multi-GiB validation tensors
    # on very large (M, N) shapes.
    cosine_rows = min(x_2d.shape[0], max(1, cosine_check_max_rows))
    if cosine_rows < x_2d.shape[0]:
        cosine_idx = torch.linspace(
            0,
            x_2d.shape[0] - 1,
            steps=cosine_rows,
            device=x_2d.device,
            dtype=torch.float32,
        ).round().to(torch.long)
        x_for_cosine = x_2d.index_select(0, cosine_idx)
    else:
        x_for_cosine = x_2d
    bf16_ref = torch.matmul(x_for_cosine, weight_bf16.t()).float()

    def _cos_sim(a, b):
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        denom = (a_flat.norm() * b_flat.norm()).clamp_min(1e-12)
        return torch.dot(a_flat, b_flat).item() / denom.item()

    is_large_shape = (in_features >= 4096) or (out_features >= 4096)

    def _cfg_id(name: str):
        m = re.fullmatch(r"cfg(\d+)", name)
        return int(m.group(1)) if m else -1

    # Heuristic pruning: focus on Nb=256-heavy configs for large shapes and
    # keep a smaller, diverse set for small shapes.
    if is_large_shape:
        allowed_cfg_ids = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            30, 31, 32, 33, 34, 42, 43, 44, 45, 46
        }
    else:
        allowed_cfg_ids = {0, 13, 14, 15, 16, 28, 38, 39, 40, 41, 47, 48, 49, 50, 51, 1, 2, 6, 22, 23}

    candidate_variants = []
    for variant_name, gemm_fn in tk_gemm_variants.items():
        cfg_id = _cfg_id(variant_name)
        if cfg_id >= 0 and cfg_id not in allowed_cfg_ids:
            continue
        candidate_variants.append((variant_name, gemm_fn))
    if not candidate_variants:
        candidate_variants = list(tk_gemm_variants.items())

    # Autotune TK GEMM config per shape.
    best_variant = None
    best_ms = float("inf")
    with torch.no_grad():
        for variant_name, gemm_fn in candidate_variants:
            tk_layer = TKFP4LinearPrequant(
                weight_bf16=weight_bf16,
                tk_nvfp4_gemm_fn=gemm_fn,
                tk_nvfp4_quantize=tk_nvfp4_quantize,
                tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
                scale_2d=scale_2d,
            )
            trial = tk_layer(x_for_cosine).float()
            cosine = _cos_sim(trial, bf16_ref)
            if not (cosine >= min_cosine_vs_bf16):
                del trial, tk_layer
                continue
            for _ in range(2):
                _ = tk_layer(x_2d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(max(1, autotune_runs)):
                _ = tk_layer(x_2d)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - start) * 1000.0 / max(1, autotune_runs)
            if elapsed_ms < best_ms:
                best_ms = elapsed_ms
                best_variant = variant_name
            del trial, tk_layer

    if best_variant is None:
        # Fallback: if all variants fail cosine gate, pick fastest without gate.
        with torch.no_grad():
            for variant_name, gemm_fn in candidate_variants:
                tk_layer = TKFP4LinearPrequant(
                    weight_bf16=weight_bf16,
                    tk_nvfp4_gemm_fn=gemm_fn,
                    tk_nvfp4_quantize=tk_nvfp4_quantize,
                    tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
                    scale_2d=scale_2d,
                )
                for _ in range(2):
                    _ = tk_layer(x_2d)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                for _ in range(max(1, autotune_runs)):
                    _ = tk_layer(x_2d)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.time() - start) * 1000.0 / max(1, autotune_runs)
                if elapsed_ms < best_ms:
                    best_ms = elapsed_ms
                    best_variant = variant_name
                del tk_layer

    assert best_variant is not None
    tk_layer = TKFP4LinearPrequant(
        weight_bf16=weight_bf16,
        tk_nvfp4_gemm_fn=tk_gemm_variants[best_variant],
        tk_nvfp4_quantize=tk_nvfp4_quantize,
        tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
        scale_2d=scale_2d,
    )

    theoretical_flops = 2 * bsz * seq_len * in_features * out_features

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = tk_layer(x_2d)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            out = tk_layer(x_2d)
            out = out + 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start
    return {
        "tflops_per_second": (theoretical_flops * num_runs) / total_time / 1e12,
        "avg_time_per_run_ms": total_time / num_runs * 1000,
        "best_variant": best_variant,
    }


def compare_tk_flashinfer_precision(
    x_bf16,
    weight_bf16,
    tk_gemm_fn,
    tk_nvfp4_quantize,
    tk_nvfp4_quantize_with_global=None,
    scale_2d=False,
    max_rows=2048,
):
    """Compare TK vs FlashInfer FP4 outputs (BF16 output tensors) for one shape."""
    x_2d = x_bf16.view(-1, x_bf16.shape[-1])
    w_t = weight_bf16.t()

    def _cos_sim(a, b):
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        denom = (a_flat.norm() * b_flat.norm()).clamp_min(1e-12)
        return torch.dot(a_flat, b_flat).item() / denom.item()

    if max_rows is not None and max_rows > 0 and x_2d.shape[0] > max_rows:
        idx = torch.linspace(
            0,
            x_2d.shape[0] - 1,
            steps=max_rows,
            device=x_2d.device,
            dtype=torch.float32,
        ).round().to(torch.long)
        x_eval = x_2d.index_select(0, idx)
    else:
        x_eval = x_2d

    with torch.no_grad():
        out_bf16 = torch.matmul(x_eval, w_t)
        # FlashInfer path
        w_global_sf = (448 * 6) / weight_bf16.float().abs().nan_to_num().max()
        w_fp4, w_scale = nvfp4_quantize(
            weight_bf16,
            w_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        x_fp4, x_scale = nvfp4_quantize(
            x_eval,
            x_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        out_scale = 1.0 / (x_global_sf * w_global_sf)
        out_fi = mm_fp4(
            x_fp4,
            w_fp4.t(),
            x_scale,
            w_scale.t(),
            out_scale,
            torch.bfloat16,
            None,
            backend="cutlass",
        )

        # TK path (selected/autotuned kernel config)
        tk_layer = TKFP4LinearPrequant(
            weight_bf16=weight_bf16,
            tk_nvfp4_gemm_fn=tk_gemm_fn,
            tk_nvfp4_quantize=tk_nvfp4_quantize,
            tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
            scale_2d=scale_2d,
        )
        out_tk = tk_layer(x_eval)

    out_fi_f32 = out_fi.float()
    out_tk_f32 = out_tk.float()
    out_bf16_f32 = out_bf16.float()
    abs_diff = (out_tk_f32 - out_fi_f32).abs()
    rel_diff = abs_diff / out_fi_f32.abs().clamp_min(1e-6)
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "max_rel": rel_diff.max().item(),
        "mean_rel": rel_diff.mean().item(),
        "cos_tk_fi": _cos_sim(out_tk_f32, out_fi_f32),
        "cos_fi_bf16": _cos_sim(out_fi_f32, out_bf16_f32),
        "cos_tk_bf16": _cos_sim(out_tk_f32, out_bf16_f32),
    }


def run_benchmark(
    num_runs=50,
    warmup_runs=5,
    tk_scale_2d=False,
    tk_autotune_runs=5,
    tk_min_cosine=0.95,
    tk_debug_mismatch=False,
    tk_debug_layer_regex="to_k|to_v",
    tk_debug_fixed_cfgs=None,
    tk_debug_full_rows=True,
):
    print("=== FP4 vs BF16 Benchmark (Weight Pre-Quantized, Activation Quant Timed) ===")

    if not torch.cuda.is_available():
        print("CUDA is required for Flashinfer FP4 benchmark, but no CUDA device is available.")
        return

    device = "cuda"
    print(f"Using device: {device}")
    # Temporarily disable all ThunderKittens benchmark paths.
    # tk_ops = _try_import_tk_nvfp4_ops()
    tk_ops = None
    tk_enabled = False
    print("TK NVFP4 extension: disabled")

    text_seq_len = 1024
    video_seq_len = 32640
    audio_seq_len = 126
    single_step_seq_len = 1

    base_configs = [
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_q"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_k"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_v"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn1.to_out"},
        {"batch_size": 1, "seq_len": text_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_k"},
        {"batch_size": 1, "seq_len": text_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_k"},
        {"batch_size": 1, "seq_len": text_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_v"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.attn2.to_out"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_q"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_k"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_v"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn1.to_out"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_q"},
        {"batch_size": 1, "seq_len": text_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_k"},
        {"batch_size": 1, "seq_len": text_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_v"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_attn2.to_out"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 2048, "name": "ltx2.blocks.32.audio_to_video_attn.to_q"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_to_video_attn.to_k"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.audio_to_video_attn.to_v"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 2048, "out_features": 4096, "name": "ltx2.blocks.32.audio_to_video_attn.to_out"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_q"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_k"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_v"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.video_to_audio_attn.to_out"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 32, "name": "ltx2.blocks.32.attn1.to_gate_logits"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 32, "name": "ltx2.blocks.32.attn2.to_gate_logits"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 32, "name": "ltx2.blocks.32.audio_attn1.to_gate_logits"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 32, "name": "ltx2.blocks.32.audio_attn2.to_gate_logits"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 32, "name": "ltx2.blocks.32.audio_to_video_attn.to_gate_logits"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 32, "name": "ltx2.blocks.32.video_to_audio_attn.to_gate_logits"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 32, "name": "ltx2.blocks.32.attn1.to_gate_compress"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 36864, "name": "ltx2.blocks.32.adaln_single.linear"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 18432, "name": "ltx2.blocks.32.audio_adaln_single.linear"},
        {"batch_size": 1, "seq_len": single_step_seq_len, "in_features": 4096, "out_features": 8192, "name": "ltx2.blocks.32.prompt_adaln_single.linear"},
        {"batch_size": 1, "seq_len": single_step_seq_len, "in_features": 2048, "out_features": 4096, "name": "ltx2.blocks.32.audio_prompt_adaln_single.linear"},
        {"batch_size": 1, "seq_len": single_step_seq_len, "in_features": 4096, "out_features": 16384, "name": "ltx2.blocks.32.av_ca_video_scale_shift_adaln_single.linear"},
        {"batch_size": 1, "seq_len": single_step_seq_len, "in_features": 2048, "out_features": 8192, "name": "ltx2.blocks.32.av_ca_audio_scale_shift_adaln_single.linear"},
        {"batch_size": 1, "seq_len": single_step_seq_len, "in_features": 4096, "out_features": 4096, "name": "ltx2.blocks.32.av_ca_a2v_gate_adaln_single.linear"},
        {"batch_size": 1, "seq_len": single_step_seq_len, "in_features": 2048, "out_features": 2048, "name": "ltx2.blocks.32.av_ca_v2a_gate_adaln_single.linear"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 4096, "out_features": 16384, "name": "ltx2.blocks.32.ffn.fc_in"},
        {"batch_size": 1, "seq_len": video_seq_len, "in_features": 16384, "out_features": 4096, "name": "ltx2.blocks.32.ffn.fc_out"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 2048, "out_features": 8192, "name": "ltx2.blocks.32.audio.ffn.fc_in"},
        {"batch_size": 1, "seq_len": audio_seq_len, "in_features": 8192, "out_features": 2048, "name": "ltx2.blocks.32.audio.ffn.fc_out"},
    ]

    def build_scaled_configs(configs, divisor):
        scaled = []
        for cfg in configs:
            cfg_scaled = dict(cfg)
            if cfg["seq_len"] == video_seq_len:
                cfg_scaled["seq_len"] = max(1, cfg["seq_len"] // divisor)
            else:
                cfg_scaled["seq_len"] = cfg["seq_len"]
            cfg_scaled["name"] = f"{cfg['name']} [seq/{divisor}]"
            scaled.append(cfg_scaled)
        return scaled

    def run_tk_fi_debug(configs, section_name):
        if not tk_enabled:
            print(f"\n[TK debug] Skipping {section_name}: TK extension unavailable.")
            return
        if tk_debug_layer_regex:
            matcher = re.compile(tk_debug_layer_regex)
            debug_configs = [cfg for cfg in configs if matcher.search(cfg["name"])]
        else:
            debug_configs = list(configs)
        if not debug_configs:
            print(f"\n[TK debug] No layers matched regex '{tk_debug_layer_regex}' in {section_name}.")
            return

        print(f"\n=== TK vs FlashInfer Debug: {section_name} ===")
        print(
            f"{'idx':>4} {'matmul':<48}{'cfg':>12}{'max_abs':>12}{'mean_abs':>12}"
            f"{'cos(TK,FI)':>12}{'cos(FI,BF16)':>14}{'cos(TK,BF16)':>14}"
            f"{'A_dec(TK/FI)':>16}{'B_dec(TK/FI)':>16}"
        )
        print(
            f"[TK debug] tk_nvfp4_quantize_with_global available: "
            f"{'yes' if tk_nvfp4_quantize_with_global is not None else 'no'}"
        )
        if tk_nvfp4_quantize_with_global is None:
            print(
                "[TK debug] Warning: TK path uses nvfp4_quantize() (computed global scale), "
                "while FI path uses fixed x_global_sf=1.0 in this script."
            )

        for i, config in enumerate(debug_configs):
            bsz = config["batch_size"]
            seq_len = config["seq_len"]
            in_features = config["in_features"]
            out_features = config["out_features"]

            torch.manual_seed(42)
            x = torch.randn(bsz, seq_len, in_features, dtype=torch.bfloat16, device=device)
            w = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)

            x_2d = x.view(-1, in_features)
            w_amax = w.float().abs().nan_to_num().max().item()
            x_amax = x_2d.float().abs().nan_to_num().max().item()
            fi_a_dec = 1.0 / float(x_global_sf.item())
            fi_b_dec = w_amax / (448.0 * 6.0)

            cfgs_to_run = tk_debug_fixed_cfgs if tk_debug_fixed_cfgs else ["cfg_default"]
            for cfg_name in cfgs_to_run:
                gemm_fn = tk_gemm_variants.get(cfg_name)
                if gemm_fn is None:
                    print(f"[TK debug] Missing cfg '{cfg_name}' for layer {config['name']}, skipping.")
                    continue
                precision = compare_tk_flashinfer_precision(
                    x,
                    w,
                    tk_gemm_fn=gemm_fn,
                    tk_nvfp4_quantize=tk_nvfp4_quantize,
                    tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
                    scale_2d=tk_scale_2d,
                    max_rows=None if tk_debug_full_rows else 2048,
                )

                tk_layer = TKFP4LinearPrequant(
                    weight_bf16=w,
                    tk_nvfp4_gemm_fn=gemm_fn,
                    tk_nvfp4_quantize=tk_nvfp4_quantize,
                    tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
                    scale_2d=tk_scale_2d,
                )
                # Force quantization once to materialize the activation global scale.
                _ = tk_layer(x_2d)
                tk_a_dec = float(tk_layer._A_sc_global.item())
                tk_b_dec = float(tk_layer.B_sc_global.item())

                print(
                    f"{i:>4} {config['name']:<48}{cfg_name:>12}"
                    f"{precision['max_abs']:>12.6f}{precision['mean_abs']:>12.6f}"
                    f"{precision['cos_tk_fi']:>12.6f}{precision['cos_fi_bf16']:>14.6f}{precision['cos_tk_bf16']:>14.6f}"
                    f"{f'{tk_a_dec:.6g}/{fi_a_dec:.6g}':>16}{f'{tk_b_dec:.6g}/{fi_b_dec:.6g}':>16}"
                )
                del tk_layer
            del x, w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"[TK debug] Note: x_amax and w_amax vary per layer; x_global_sf fixed at {float(x_global_sf.item()):.6g}. "
            f"Last layer x_amax={x_amax:.6g}"
        )

    def run_section(configs, section_name, tk_scale_2d=False):
        fp4_faster_layers = []
        tk_faster_than_flashinfer = []
        flashinfer_faster_than_tk = []
        precision_rows = []

        print(f"\n=== {section_name} ===")
        print("matmul-performance (weight quant excluded for FP4):")
        if tk_enabled:
            print(
                f"{'idx':>4} {'matmul':<48}{'N':>12}{'K':>12}"
                f"{'Torch-BF16':>14}{'Flashinfer-fp4':>16}{'TK-fp4':>12}{'TK_vs_FI':>12}{'TK_cfg':>12}"
            )
        else:
            print(
                f"{'idx':>4} {'matmul':<48}{'N':>12}{'K':>12}"
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
            tk_results = None
            if tk_enabled:
                tk_results = benchmark_tk_fp4_prequant_weight_tflops(
                    x,
                    w,
                    tk_gemm_variants=tk_gemm_variants,
                    tk_nvfp4_quantize=tk_nvfp4_quantize,
                    tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs,
                    scale_2d=tk_scale_2d,
                    autotune_runs=tk_autotune_runs,
                    min_cosine_vs_bf16=tk_min_cosine,
                )

            tk_tflops = tk_results["tflops_per_second"] if tk_results else float("nan")
            if tk_results:
                fi_tflops = fp4_results["tflops_per_second"]
                tk_cfg = tk_results.get("best_variant", "n/a")
                if fi_tflops <= 0 or tk_tflops <= 0:
                    ratio_text = "n/a"
                else:
                    ratio = tk_tflops / fi_tflops
                    faster_tag = "TK" if ratio > 1.0 else "FI"
                    ratio_text = f"{faster_tag}:{ratio:.3f}x" if ratio > 1.0 else f"{faster_tag}:{1.0/ratio:.3f}x"
                if fi_tflops > 0 and tk_tflops > fi_tflops:
                    tk_faster_than_flashinfer.append(
                        {
                            "name": config["name"],
                            "tk_tflops": tk_tflops,
                            "fi_tflops": fi_tflops,
                            "bf16_tflops": bf16_results["tflops_per_second"],
                            "speedup": tk_tflops / fi_tflops,
                        }
                    )
                elif tk_tflops > 0 and fi_tflops > 0:
                    flashinfer_faster_than_tk.append(
                        {
                            "name": config["name"],
                            "tk_tflops": tk_tflops,
                            "fi_tflops": fi_tflops,
                            "bf16_tflops": bf16_results["tflops_per_second"],
                            "speedup": fi_tflops / tk_tflops,
                        }
                    )
                precision = compare_tk_flashinfer_precision(
                    x,
                    w,
                    tk_gemm_fn=tk_gemm_variants[tk_cfg],
                    tk_nvfp4_quantize=tk_nvfp4_quantize,
                    tk_nvfp4_quantize_with_global=tk_nvfp4_quantize_with_global,
                    scale_2d=tk_scale_2d,
                )
                precision_rows.append(
                    {
                        "name": config["name"],
                        "tk_cfg": tk_cfg,
                        **precision,
                    }
                )
            else:
                ratio_text = "n/a"
                tk_cfg = "n/a"

            if tk_enabled:
                print(
                    f"{i:>4} {config['name']:<48}"
                    f"{in_features:>12.1f}{out_features:>12.1f}"
                    f"{bf16_results['tflops_per_second']:>14.6f}{fp4_results['tflops_per_second']:>16.6f}"
                    f"{tk_tflops:>12.6f}{ratio_text:>12}{tk_cfg:>12}"
                )
            else:
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

        if tk_enabled:
            tk_faster_and_beats_bf16 = [
                item for item in tk_faster_than_flashinfer if item["tk_tflops"] > item["bf16_tflops"]
            ]
            flashinfer_faster_and_beats_bf16 = [
                item for item in flashinfer_faster_than_tk if item["fi_tflops"] > item["bf16_tflops"]
            ]
            print("\nTK vs Flashinfer per-shape summary:")
            print(
                "  - Entries shown only when the winner also beats BF16 on that shape"
            )
            print(f"  - TK faster on {len(tk_faster_and_beats_bf16)}/{len(configs)} shapes")
            for item in tk_faster_and_beats_bf16:
                print(
                    f"    TK faster: {item['name']}: "
                    f"{item['speedup']:.4f}x "
                    f"(TK={item['tk_tflops']:.6f}, Flashinfer={item['fi_tflops']:.6f}, BF16={item['bf16_tflops']:.6f})"
                )
            for item in flashinfer_faster_and_beats_bf16:
                print(
                    f"    Flashinfer faster: {item['name']}: "
                    f"{item['speedup']:.4f}x "
                    f"(Flashinfer={item['fi_tflops']:.6f}, TK={item['tk_tflops']:.6f}, BF16={item['bf16_tflops']:.6f})"
                )
            print("\nTK vs Flashinfer precision (winner TK_cfg):")
            print(
                f"{'idx':>4} {'matmul':<48}{'cfg':>12}{'max_abs':>12}{'mean_abs':>12}"
                f"{'cos(TK,FI)':>12}{'cos(FI,BF16)':>14}{'cos(TK,BF16)':>14}"
            )
            for i, row in enumerate(precision_rows):
                print(
                    f"{i:>4} {row['name']:<48}{row['tk_cfg']:>12}"
                    f"{row['max_abs']:>12.6f}{row['mean_abs']:>12.6f}"
                    f"{row['cos_tk_fi']:>12.6f}{row['cos_fi_bf16']:>14.6f}{row['cos_tk_bf16']:>14.6f}"
                )

    run_section(base_configs, "Original Sequence Length", tk_scale_2d=tk_scale_2d)
    run_section(build_scaled_configs(base_configs, 4), "Sequence Length Divided By 4", tk_scale_2d=tk_scale_2d)
    if tk_debug_mismatch:
        run_tk_fi_debug(base_configs, "Original Sequence Length")
        run_tk_fi_debug(build_scaled_configs(base_configs, 4), "Sequence Length Divided By 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=50, help="Timed iterations per shape")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warmup iterations per shape")
    parser.add_argument("--tk-scale-2d", action="store_true", help="Use TK 2D scaling mode in nvfp4_quantize")
    parser.add_argument("--tk-autotune-runs", type=int, default=5, help="Autotune iterations per TK config per shape")
    parser.add_argument("--tk-min-cosine", type=float, default=0.95, help="Reject TK autotune configs below cosine vs BF16")
    parser.add_argument("--tk-debug-mismatch", action="store_true", help="Run focused TK-vs-FlashInfer debug compare with scale dumps")
    parser.add_argument("--tk-debug-layer-regex", type=str, default="to_k|to_v", help="Regex filter for debug layers")
    parser.add_argument("--tk-debug-fixed-cfgs", type=str, default="", help="Comma-separated TK cfg names to force in debug mode (e.g. cfg2,cfg6,cfg27)")
    parser.add_argument("--tk-debug-full-rows", action="store_true", help="Use full rows (no downsample) in TK debug precision compare")
    args = parser.parse_args()

    tk_debug_fixed_cfgs = [x.strip() for x in args.tk_debug_fixed_cfgs.split(",") if x.strip()]

    run_benchmark(
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        tk_scale_2d=args.tk_scale_2d,
        tk_autotune_runs=args.tk_autotune_runs,
        tk_min_cosine=args.tk_min_cosine,
        tk_debug_mismatch=args.tk_debug_mismatch,
        tk_debug_layer_regex=args.tk_debug_layer_regex,
        tk_debug_fixed_cfgs=tk_debug_fixed_cfgs,
        tk_debug_full_rows=args.tk_debug_full_rows,
    )

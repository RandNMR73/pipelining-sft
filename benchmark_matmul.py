#!/usr/bin/env python3
"""
Benchmark script to measure TFLOPs for matrix multiplication operations only.
Compares FP8 vs BF16 matrix multiplication performance.
"""

import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np

# Add the models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.deepseek_v3.fp8_layers import per_token_cast_to_fp8, per_block_cast_to_fp8
import deep_gemm

def benchmark_matmul_tflops(x, weight, num_runs=100, warmup_runs=10, operation_name="MatMul"):
    """
    Benchmark the TFLOPs of a matrix multiplication operation.
    
    Args:
        x: Input tensor
        weight: Weight tensor
        num_runs: Number of runs for averaging
        warmup_runs: Number of warmup runs
        operation_name: Name of the operation being benchmarked
        
    Returns:
        dict: Dictionary containing timing and TFLOPs information
    """
    device = x.device
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    
    # Calculate theoretical TFLOPs for one forward pass
    # For matrix multiplication: 2 * batch_size * seq_len * in_features * out_features
    theoretical_flops = 2 * batch_size * seq_len * in_features * out_features
    theoretical_tflops = theoretical_flops / 1e12
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            if operation_name == "FP8 MatMul":
                # FP8 path: cast to FP8 then use DeepGEMM
                x_reshaped = x.view(-1, in_features)  # (batch_size * seq_len, in_features)
                x_fp8, x_scales = per_token_cast_to_fp8(x_reshaped)
                weight_fp8, weight_scales = per_block_cast_to_fp8(weight)
                
                # Prepare tensors for DeepGEMM - match original implementation exactly
                x_fp8_aligned = (x_fp8, deep_gemm.get_mn_major_tma_aligned_tensor(x_scales))
                weight_fp8_aligned = weight_fp8  # Don't wrap weight scales in get_mn_major_tma_aligned_tensor
                
                # Create output tensor with correct dimensions
                out = torch.zeros((x_reshaped.shape[0], out_features), device=x.device, dtype=x.dtype)
                deep_gemm.fp8_gemm_nt(x_fp8_aligned, weight_fp8_aligned, out)
            else:
                # BF16 path: direct multiplication
                _ = torch.matmul(x.view(-1, in_features), weight)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            if operation_name == "FP8 MatMul":
                # FP8 path: cast to FP8 then use DeepGEMM
                x_reshaped = x.view(-1, in_features)  # (batch_size * seq_len, in_features)
                x_fp8, x_scales = per_token_cast_to_fp8(x_reshaped)
                weight_fp8, weight_scales = per_block_cast_to_fp8(weight)
                
                # Prepare tensors for DeepGEMM - match original implementation exactly
                x_fp8_aligned = (x_fp8, deep_gemm.get_mn_major_tma_aligned_tensor(x_scales))
                weight_fp8_aligned = weight_fp8  # Don't wrap weight scales in get_mn_major_tma_aligned_tensor
                
                # Create output tensor with correct dimensions
                out = torch.zeros((x_reshaped.shape[0], out_features), device=x.device, dtype=x.dtype)
                deep_gemm.fp8_gemm_nt(x_fp8_aligned, weight_fp8_aligned, out)
            else:
                # BF16 path: direct multiplication
                _ = torch.matmul(x.view(-1, in_features), weight)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate timing and TFLOPs
    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs
    flops_per_second = (theoretical_flops * num_runs) / total_time
    tflops_per_second = flops_per_second / 1e12
    
    return {
        'theoretical_tflops': theoretical_tflops,
        'avg_time_per_run_ms': avg_time_per_run * 1000,
        'total_time_s': total_time,
        'tflops_per_second': tflops_per_second,
        'flops_per_second': flops_per_second,
        'num_runs': num_runs,
        'input_shape': x.shape,
        'output_shape': (batch_size, seq_len, out_features)
    }

def benchmark_fp8_vs_bf16_matmul():
    """Benchmark TFLOPs performance of FP8 vs BF16 matrix multiplication."""
    print("\n=== Benchmarking FP8 vs BF16 Matrix Multiplication TFLOPs ===")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Using device: {device}")
        
        # Test configurations matching the benchmark data
        configs = [
            {'batch_size': 1, 'seq_len': 32000, 'in_features': 4096, 'out_features': 4096, 'name': '4K×4K'},
            {'batch_size': 1, 'seq_len': 32000, 'in_features': 8192, 'out_features': 8192, 'name': '8K×8K'},
            {'batch_size': 1, 'seq_len': 32000, 'in_features': 12288, 'out_features': 12288, 'name': '12K×12K'},
            {'batch_size': 1, 'seq_len': 32000, 'in_features': 16384, 'out_features': 16384, 'name': '16K×16K'},
        ]
        
        results = []
        
        # Print header
        print("\nmatmul-performance:")
        print(f"{'':>8}{'N':>12}{'K':>12}{'BF16-MatMul':>14}{'FP8-MatMul':>12}")
        
        for i, config in enumerate(configs):
            # Create test inputs
            x = torch.randn(
                config['batch_size'], 
                config['seq_len'], 
                config['in_features'], 
                dtype=torch.bfloat16, 
                device=device
            )
            
            weight = torch.randn(
                config['out_features'],
                config['in_features'],
                dtype=torch.bfloat16,
                device=device
            )
            
            print(f"Benchmarking {config['name']} configuration...")
            
            # Benchmark BF16 matrix multiplication
            bf16_results = benchmark_matmul_tflops(
                x, weight, 
                num_runs=50, 
                warmup_runs=5, 
                operation_name="BF16 MatMul"
            )
            
            # Benchmark FP8 matrix multiplication
            fp8_results = benchmark_matmul_tflops(
                x, weight, 
                num_runs=50, 
                warmup_runs=5, 
                operation_name="FP8 MatMul"
            )
            
            # Calculate speed-up as FP8 throughput divided by BF16 throughput
            speedup = fp8_results['tflops_per_second'] / bf16_results['tflops_per_second']
            
            # Store results
            config_result = {
                'config': config,
                'fp8': fp8_results,
                'bf16': bf16_results,
                'speedup': speedup
            }
            results.append(config_result)
            
            # Print results in the specified format
            print(f"{i:>8}{config['in_features']:>12.1f}{config['out_features']:>12.1f}{bf16_results['tflops_per_second']:>14.6f}{fp8_results['tflops_per_second']:>12.6f}")
        
        # Print summary
        print("\n" + "="*60)
        print("Matrix Multiplication Benchmark Summary")
        print("="*60)
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"Average speedup (FP8 / BF16): {avg_speedup:.2f}x")
        
        if avg_speedup > 1:
            print("🎯 FP8 matrix multiplication is faster on average!")
        else:
            print("⚠️  BF16 matrix multiplication is faster on average")
        
        print("\nNote: This benchmark measures raw matrix multiplication operations.")
        print("For production use, optimized FP8 kernels (DeepGEMM) would provide better performance.")
        
        return True
        
    except Exception as e:
        print(f"✗ Matrix multiplication benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the matrix multiplication benchmark."""
    print("Matrix Multiplication TFLOPs Benchmark")
    print("=" * 50)
    
    try:
        result = benchmark_fp8_vs_bf16_matmul()
        if result:
            print("\n✅ Benchmark completed successfully!")
        else:
            print("\n❌ Benchmark failed!")
    except Exception as e:
        print(f"\n❌ Benchmark crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
"""
FP8 training layers for DeepSeek V3 using DeepGEMM backend.
Maintains FP32 master weights while performing computations in FP8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import deep_gemm
from deep_gemm import ceil_div, get_mn_major_tma_aligned_tensor
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8
from models.deepseek_v3.fp8_layers_triton import per_token_cast_to_fp8_triton, per_block_cast_to_fp8_triton

block_size = 128

# def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     assert x.dim() == 2
#     m, n = x.shape
#     pad_size = (128 - (n % 128)) % 128
#     x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
#     x_view = x.view(m, -1, 128)
#     x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
#     fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
#     return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

# def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     assert x.dim() == 2
#     m, n = x.shape
#     x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
#     x_padded[:m, :n] = x
#     x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
#     x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
#     x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
#     return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def functional_fp8_linear(x: torch.Tensor, weight, bias=None) -> torch.Tensor:
    """
    Args:
        x: input tensor, shape (B, N)
        weight: weight tensor, shape (M, N)

    Returns:
        torch.Tensor: output from custom FP8 GEMM
    """
    assert bias is None, "We don't expect bias"

    # Your weight must already be quantized to FP8 and carry .scale
    # if not hasattr(weight, 'scale'):
    #     raise RuntimeError("weight must have .scale attribute for FP8 GEMM")

    return FP8Linear.apply(x, weight)

class FP8Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None):
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to fp8 linear"

        shape = x.shape

        x = x.view(-1, shape[-1])

        x_fp8 = per_token_cast_to_fp8(x, use_ue8m0=True)
        # x_fp8 = per_token_cast_to_fp8_triton(x)
        x_fp8 = (x_fp8[0], get_mn_major_tma_aligned_tensor(x_fp8[1]))

        weight_fp8 = per_block_cast_to_fp8(weight, use_ue8m0=True)
        # weight_fp8 = per_block_cast_to_fp8_triton(weight)
        ctx.save_for_backward(x, weight)
        out_dim = weight.shape[0]
        # flattened
        out = torch.zeros((x.shape[0], out_dim), device=x.device, dtype=x.dtype)
        deep_gemm.fp8_gemm_nt(x_fp8, weight_fp8, out, disable_ue8m0_cast=True)
        if len(shape) == 3:
            out = out.view(shape[0], shape[1], out_dim)
        return out  

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_weight = grad_bias = None

        shape = grad_output.shape
        grad_output = grad_output.view(-1, shape[-1])

        x, weight = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            # 1. d_weight
            # the scaling has to be done on the channel dim so we cast to fp8 with l, c
            # then transpose them, seems to be more proper, but i will use the common method since people are doing it
            dy_fp8 = per_token_cast_to_fp8(grad_output.t().contiguous(), use_ue8m0=False)  # c, l
            x_fp8 = per_token_cast_to_fp8(x.t().contiguous())

            dy_fp8 = (dy_fp8[0], get_mn_major_tma_aligned_tensor(dy_fp8[1]))
            x_fp8 = (x_fp8[0], get_mn_major_tma_aligned_tensor(x_fp8[1]))

            grad_weight = torch.zeros_like(weight, dtype=torch.float32)

            deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(dy_fp8, x_fp8, grad_weight)

            # grad_weight = None

        # 2. d_input
        if ctx.needs_input_grad[0]:
            dy_fp8 = per_token_cast_to_fp8(grad_output.contiguous())
            dy_fp8 = (dy_fp8[0], get_mn_major_tma_aligned_tensor(dy_fp8[1]))
            weight_fp8 = per_block_cast_to_fp8(weight.t().contiguous(), use_ue8m0=False)
            # weight_fp8 = (weight, weight.scale)
            # weight_fp8 = (weight_fp8[0].t().contiguous(), weight_fp8[1].t().contiguous())

            # ref_grad_input = grad_output.float() @ weight_dequant(weight, weight.scale).float()

            grad_input = torch.zeros_like(x)
            deep_gemm.fp8_gemm_nt(dy_fp8, weight_fp8, grad_input)

            if len(shape) == 3:
                in_dim = weight.shape[1]
                grad_input = grad_input.view(shape[0], shape[1], in_dim)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Linear):
    """
    Custom linear layer. 

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert not bias, f"we don't support bias!"
        super().__init__(in_features, out_features, bias, dtype=torch.float32)

        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """

        if self.weight.element_size() != 2:
            return functional_fp8_linear(
                x, self.weight, self.bias
            )
        else:
            return F.linear(x, self.weight, self.bias)


def convert_linear_to_fp8(module: nn.Module, fp8_enabled: bool = True, prefix: str = "") -> nn.Module:
    """
    Convert nn.Linear layers to FP8 Linear layers following DeepSeek V3 FP8 training guidelines.
    
    According to the paper, these modules should NOT be converted to FP8:
    - Embedding layers (embed_tokens)
    - Output head (lm_head) 
    - MoE gating modules (gate)
    - Normalization operators (RMSNorm, LayerNorm)
    - Attention operators (q_norm, kv_norm, etc.)
    
    Args:
        module: Module to convert
        fp8_enabled: Whether to enable FP8 computation
        
    Returns:
        Module with converted layers
    """
    # Import here to avoid circular imports
    from .layers import RMSNorm, Gate, MLP, Expert
    from .attention import MLA
    
    # Define modules that should NOT be converted to FP8
    PRESERVE_PRECISION_MODULES = (
        nn.Embedding,           # Embedding layers
        RMSNorm,               # Normalization layers
        Gate,                  # MoE gating modules
    )
    
    # Define module/parameter names that should preserve precision
    PRESERVE_PRECISION_NAMES = {
        'embed',               # Embedding tokens
        'embed_tokens',        # HuggingFace style embedding
        'head',               # Output head  
        'gate',               # MoE gate weights
        'norm',               # Normalization weights
        'q_norm',             # Query normalization in attention
        'kv_norm',            # Key-value normalization in attention
        'attn_norm',          # Attention normalization
        'ffn_norm',           # FFN normalization
    }
    
    def should_preserve_precision(name: str, module_instance: nn.Module) -> bool:
        """Check if a module should preserve its original precision."""
        # Check by module type
        if isinstance(module_instance, PRESERVE_PRECISION_MODULES):
            return True
            
        # Check by name patterns
        name_lower = name.lower()
        for preserve_name in PRESERVE_PRECISION_NAMES:
            if preserve_name in name_lower:
                return True
                
        return False
    
    # Convert nn.Linear layers to FP8 Linear layers
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear):
            # Check if this layer should preserve precision
            if should_preserve_precision(name, child):
                print(f"Preserving precision for {full_name} (type: {type(child).__name__})")
                continue
                
            # Create new FP8 Linear layer (no bias support)
            if child.bias is not None:
                print(f"Warning: FP8 Linear doesn't support bias, skipping bias for {full_name}")
                
            fp8_layer = Linear(
                child.in_features,
                child.out_features,
                bias=False  # FP8 Linear doesn't support bias
            ).to(device=child.weight.device, dtype=torch.float32)
            
            # Copy weights from original layer
            with torch.no_grad():
                fp8_layer.weight.copy_(child.weight.float())
            
            #print(f"Setting FP8 layer for: {full_name}")
            setattr(module, name, fp8_layer)
            #print(f"Converted {full_name} to FP8 Linear: {child.in_features} -> {child.out_features}")
            
        elif isinstance(child, MLP):
            # Convert MLP layers to use FP8 Linear
            #print(f"Converting MLP {full_name} to use FP8 Linear layers")
            convert_linear_to_fp8(child, fp8_enabled, full_name)
            
        elif isinstance(child, Expert):
            # Convert Expert layers to use FP8 Linear
            #print(f"Converting Expert {full_name} to use FP8 Linear layers")
            convert_linear_to_fp8(child, fp8_enabled, full_name)
            
        elif isinstance(child, MLA):
            # Convert attention linear layers (but preserve normalization)
            #print(f"Converting MLA {full_name} attention projections to FP8")
            convert_linear_to_fp8(child, fp8_enabled, full_name)
            
        else:
            # Recursively convert child modules
            convert_linear_to_fp8(child, fp8_enabled, full_name)
    
    return module


def count_fp8_layers(module: nn.Module) -> dict:
    """
    Count the number of FP8 vs regular layers in the model.
    
    Args:
        module: Module to analyze
        
    Returns:
        Dictionary with counts of different layer types
    """
    from .layers import MLP, Expert
    
    counts = {
        'fp8_linear': 0,
        'regular_linear': 0,
        'mlp_layers': 0,
        'expert_layers': 0,
        'preserved_layers': 0
    }
    
    PRESERVE_PRECISION_MODULES = (
        nn.Embedding,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
    )
    
    for name, child in module.named_modules():
        if isinstance(child, Linear):
            counts['fp8_linear'] += 1
        elif isinstance(child, nn.Linear):
            counts['regular_linear'] += 1
        elif isinstance(child, MLP):
            counts['mlp_layers'] += 1
        elif isinstance(child, Expert):
            counts['expert_layers'] += 1
        elif isinstance(child, PRESERVE_PRECISION_MODULES):
            counts['preserved_layers'] += 1
    
    return counts


def print_fp8_conversion_summary(model: nn.Module, rank: int = 0):
    """
    Print a summary of FP8 layer conversion.
    
    Args:
        model: The converted model
        rank: Process rank (only rank 0 prints)
    """
    if rank != 0:
        return
        
    counts = count_fp8_layers(model)
    
    print("="*60)
    print("FP8 Layer Conversion Summary")
    print("="*60)
    print(f"FP8 Linear layers:        {counts['fp8_linear']}")
    print(f"Regular Linear layers:    {counts['regular_linear']}")
    print(f"MLP layers:               {counts['mlp_layers']}")
    print(f"Expert layers:            {counts['expert_layers']}")
    print(f"Preserved layers:         {counts['preserved_layers']}")
    
    total_fp8 = counts['fp8_linear']
    total_regular = counts['regular_linear']
    total_converted = total_fp8 + counts['mlp_layers'] + counts['expert_layers']
    
    print(f"\nConversion Statistics:")
    print(f"  Total FP8 conversions:   {total_converted}")
    print(f"  Preserved precision:     {total_regular + counts['preserved_layers']}")
    
    if total_regular > 0:
        print(f"\n⚠️  {total_regular} Linear layers were not converted to FP8")
        print("   This is expected for embedding, head, gating, and normalization layers")
    else:
        print(f"\n✅ All eligible Linear layers converted to FP8!")
    
    print("="*60)

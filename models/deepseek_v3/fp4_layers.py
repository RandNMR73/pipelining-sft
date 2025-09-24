"""
FP8 training layers for DeepSeek V3 using DeepGEMM backend.
Maintains FP32 master weights while performing computations in FP8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from flashinfer import nvfp4_quantize, mm_fp4, SfLayout

def functional_fp4_linear(x: torch.Tensor, weight, bias=None) -> torch.Tensor:
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

    return FP4Linear.apply(x, weight)

class FP4Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None):
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to fp4 linear"

        shape = x.shape

        x = x.view(-1, shape[-1])
        
        x_global_sf = (448 * 6) / x.float().abs().nan_to_num().max();
        x_fp4, x_scale = nvfp4_quantize(x, x_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        weight_global_sf = (448 * 6) / weight.float().abs().nan_to_num().max();
        fp4_w, fp4_s = nvfp4_quantize(weight, weight_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        out = mm_fp4(x_fp4, fp4_w.T, x_scale, fp4_s.T, 1.0/(x_global_sf * weight_global_sf), torch.bfloat16, None, backend='cutlass')

        ctx.save_for_backward(x, weight)
        out_dim = weight.shape[0]

        if len(shape) == 3:
            out = out.view(shape[0], shape[1], out_dim)
        return out  

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_weight = grad_bias = None
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
            return functional_fp4_linear(
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

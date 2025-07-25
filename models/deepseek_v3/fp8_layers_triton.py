from typing import Tuple

import deep_gemm
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from deep_gemm import calc_diff, ceil_div, get_col_major_tma_aligned_tensor


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128, dtype=None) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype or torch.get_default_dtype())

    def get_grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    grid = get_grid
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


def functional_fp8_linear(x: torch.Tensor, weight, bias=None) -> torch.Tensor:
    """
    Args:
        x: input tensor, shape (B, N)
        layer: nn.Linear instance

    Returns:
        torch.Tensor: output from custom FP8 GEMM
    """
    assert bias is None, "We don't expect bias"

    return FP8Linear.apply(x, weight)


class FP8Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None):
        # Need contiguous tensors for collectives.
        assert x.dtype == torch.bfloat16, f"only allow bf16 inputs to fp8 linear"

        shape = x.shape

        x = x.view(-1, shape[-1])

        # x_fp8 = per_token_cast_to_fp8(x)
        x_fp8 = per_token_cast_to_fp8_triton(x)
        x_fp8 = (x_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(x_fp8[1].contiguous()))

        weight_fp8 = per_block_cast_to_fp8_triton(weight)
        ctx.save_for_backward(x, weight)
        out_dim = weight.shape[0]
        # flattened
        out = torch.zeros((x.shape[0], out_dim), device=x.device, dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, weight_fp8, out)
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
            dy_fp8 = per_token_cast_to_fp8_triton(grad_output.t().contiguous())  # c, l
            x_fp8 = per_token_cast_to_fp8_triton(x.t().contiguous())
            # dy_fp8 = per_token_cast_to_fp8(grad_output.t().contiguous())  # c, l
            # x_fp8 = per_token_cast_to_fp8(x.t().contiguous())
            
            # dy_fp8 = (dy_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(dy_fp8[1].contiguous()))
            # x_fp8 = (x_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(x_fp8[1].contiguous()))


            grad_weight = torch.zeros_like(weight, dtype=torch.float32)

            deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(dy_fp8, x_fp8, grad_weight)

        # 2. d_input
        if ctx.needs_input_grad[0]:
            dy_fp8 = per_token_cast_to_fp8_triton(grad_output.contiguous())
            # dy_fp8 = per_token_cast_to_fp8(grad_output.contiguous())
            dy_fp8 = (dy_fp8[0].contiguous(), get_col_major_tma_aligned_tensor(dy_fp8[1].contiguous()))
            weight_fp8 = per_block_cast_to_fp8_triton(weight.t().contiguous())

            grad_input = torch.zeros_like(x)
            deep_gemm.gemm_fp8_fp8_bf16_nt(dy_fp8, weight_fp8, grad_input)

            if len(shape) == 3:
                in_dim = weight.shape[1]
                grad_input = grad_input.view(shape[0], shape[1], in_dim)

        return grad_input, grad_weight, grad_bias


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

@triton.jit
def _fp8_cast_kernel(
    x_ptr,                      # Input pointer
    out_ptr,                    # Output FP8 pointer  
    scale_ptr,                  # Scale pointer
    M,                          # Number of rows
    N,                          # Number of columns (original, unpadded)
    N_padded,                   # Number of columns after padding
    stride_x_m,                 # Stride for x in M dimension
    stride_x_n,                 # Stride for x in N dimension
    stride_out_m,               # Stride for output in M dimension
    stride_out_n,               # Stride for output in N dimension
    stride_scale_m,             # Stride for scale in M dimension
    stride_scale_n,             # Stride for scale in N dimension (between chunks)
    BLOCK_SIZE: tl.constexpr,   # Size of each chunk (128)
):
    # Get the row we're processing
    row_idx = tl.program_id(0)
    
    # Calculate number of chunks
    num_chunks = (N_padded + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each chunk in the row
    for chunk_idx in range(num_chunks):
        # Calculate the starting column for this chunk
        col_start = chunk_idx * BLOCK_SIZE
        
        # Create the column indices for this chunk
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements (not padding)
        mask = cols < N
        
        # Calculate pointers for this chunk
        x_ptrs = x_ptr + row_idx * stride_x_m + cols * stride_x_n
        out_ptrs = out_ptr + row_idx * stride_out_m + cols * stride_out_n
        
        # Load the chunk (use 0.0 for padding)
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute absolute values
        x_abs = tl.abs(x_chunk)
        
        # Find maximum in this chunk
        # Note: tl.max returns a scalar when axis=0 is used on a 1D tensor
        amax = tl.max(x_abs, axis=0)
        
        # Clamp to avoid division by zero
        amax = tl.maximum(amax, 1e-4)
        
        # Compute and store scale (this is what gets stored, not used for scaling)
        scale = amax / 448.0
        scale_ptr_loc = scale_ptr + row_idx * stride_scale_m + chunk_idx * stride_scale_n
        tl.store(scale_ptr_loc, scale)
        
        # Scale the values (multiply by 448.0 / amax, not divide by scale!)
        scale_factor = 448.0 / amax
        x_scaled = x_chunk * scale_factor
        
        # Cast to FP8 (correct way in Triton)
        x_fp8 = x_scaled.to(tl.float8e4nv)
        
        # Store the FP8 values
        tl.store(out_ptrs, x_fp8, mask=mask)


def per_token_cast_to_fp8_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Correct implementation of per-token FP8 casting that matches the original PyTorch version.
    
    Args:
        x: Input tensor of shape (M, N)
    
    Returns:
        x_fp8: FP8 quantized tensor of shape (M, N)
        scales: Scale factors of shape (M, num_chunks) where num_chunks = ceil(N/128)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape
    
    # Calculate padding
    BLOCK_SIZE = 128
    pad_size = (BLOCK_SIZE - (N % BLOCK_SIZE)) % BLOCK_SIZE
    N_padded = N + pad_size
    num_chunks = N_padded // BLOCK_SIZE
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Allocate PADDED output tensor (like PyTorch version)
    x_fp8_padded = torch.empty((M, N_padded), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((M, num_chunks), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one thread block per row
    grid = (M,)
    _fp8_cast_kernel[grid](
        x,
        x_fp8_padded,  # Use padded tensor
        scales,
        M,
        N,
        N_padded,
        x.stride(0),
        x.stride(1),
        x_fp8_padded.stride(0),
        x_fp8_padded.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,  # Tune based on your GPU
    )
    return x_fp8_padded[:, :N], scales


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

@triton.jit
def _fp8_per_block_cast_kernel(
    x_ptr,                      # Input pointer
    out_ptr,                    # Output FP8 pointer  
    scale_ptr,                  # Scale pointer
    M,                          # Number of rows (original)
    N,                          # Number of columns (original)
    M_padded,                   # Number of rows after padding to multiple of 128
    N_padded,                   # Number of columns after padding to multiple of 128
    stride_x_m,                 # Stride for x in M dimension
    stride_x_n,                 # Stride for x in N dimension
    stride_out_m,               # Stride for output in M dimension
    stride_out_n,               # Stride for output in N dimension
    stride_scale_m,             # Stride for scale in M dimension (block row)
    stride_scale_n,             # Stride for scale in N dimension (block col)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    # Get the 128x128 block we're processing
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    # Calculate the starting indices for this block
    m_start = block_id_m * BLOCK_M
    n_start = block_id_n * BLOCK_N
    
    # Initialize max value for this block
    block_max = 0.0
    
    # Process the block in smaller chunks to find the maximum
    # We'll process in 32x32 tiles for efficiency
    TILE_M: tl.constexpr = 32
    TILE_N: tl.constexpr = 32
    
    # First pass: find the maximum absolute value in the entire 128x128 block
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            # Create row and column offsets for this tile
            rm = tl.arange(0, TILE_M)
            rn = tl.arange(0, TILE_N)
            
            # Calculate actual indices
            rows = m_start + tile_m + rm[:, None]
            cols = n_start + tile_n + rn[None, :]
            
            # Create mask for valid elements
            mask = (rows < M) & (cols < N)
            
            # Calculate pointers
            ptrs = x_ptr + rows * stride_x_m + cols * stride_x_n
            
            # Load tile data
            tile_data = tl.load(ptrs, mask=mask, other=0.0)
            
            # Find max absolute value in this tile
            tile_abs = tl.abs(tile_data)
            tile_max = tl.max(tile_abs)
            
            # Update block maximum
            block_max = tl.maximum(block_max, tile_max)
    
    # Clamp to avoid division by zero
    block_max = tl.maximum(block_max, 1e-4)
    
    # Compute scale factor
    scale = block_max / 448.0
    scale_factor = 448.0 / block_max
    
    # Store the scale for this block
    scale_ptr_loc = scale_ptr + block_id_m * stride_scale_m + block_id_n * stride_scale_n
    tl.store(scale_ptr_loc, scale)
    
    # Second pass: scale and convert to FP8
    for tile_m in range(0, BLOCK_M, TILE_M):
        for tile_n in range(0, BLOCK_N, TILE_N):
            # Create row and column offsets for this tile
            rm = tl.arange(0, TILE_M)
            rn = tl.arange(0, TILE_N)
            
            # Calculate actual indices
            rows = m_start + tile_m + rm[:, None]
            cols = n_start + tile_n + rn[None, :]
            
            # Create mask for valid elements
            mask = (rows < M) & (cols < N)
            
            # Calculate pointers
            in_ptrs = x_ptr + rows * stride_x_m + cols * stride_x_n
            out_ptrs = out_ptr + rows * stride_out_m + cols * stride_out_n
            
            # Load, scale, and convert tile data
            tile_data = tl.load(in_ptrs, mask=mask, other=0.0)
            tile_scaled = tile_data * scale_factor
            tile_fp8 = tile_scaled.to(tl.float8e4nv)
            
            # Store FP8 data
            tl.store(out_ptrs, tile_fp8, mask=mask)


def per_block_cast_to_fp8_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of per-block FP8 casting that matches the original PyTorch version.
    
    Each 128x128 block gets its own scale factor based on the maximum absolute value
    in that block.
    
    Args:
        x: Input tensor of shape (M, N)
    
    Returns:
        x_fp8: FP8 quantized tensor of shape (M, N)
        scales: Scale factors of shape (num_block_rows, num_block_cols)
                where num_block_rows = ceil(M/128), num_block_cols = ceil(N/128)
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape
    
    # Calculate padded dimensions
    BLOCK_SIZE = 128
    M_padded = ceil_div(M, BLOCK_SIZE) * BLOCK_SIZE
    N_padded = ceil_div(N, BLOCK_SIZE) * BLOCK_SIZE
    num_block_rows = M_padded // BLOCK_SIZE
    num_block_cols = N_padded // BLOCK_SIZE
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Allocate output tensors
    x_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((num_block_rows, num_block_cols), dtype=torch.float32, device=x.device)
    
    # Launch kernel - one thread block per 128x128 block
    grid = (num_block_rows, num_block_cols)
    _fp8_per_block_cast_kernel[grid](
        x,
        x_fp8,
        scales,
        M,
        N,
        M_padded,
        N_padded,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        scales.stride(0),
        scales.stride(1),
        num_warps=8,
    )
    
    return x_fp8, scales


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

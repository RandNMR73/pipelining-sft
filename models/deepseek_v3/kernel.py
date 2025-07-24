import torch
import triton
import triton.language as tl
import math

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

def compute_full_scaling_and_weights(weight, key=None, block_size=128):
    """
    considering x * s here probably is just blocksize * s a scaling?
    the process is you allocate a buffer in fp8 (float8_e4m3fn)
    then scale (single scalar) =  fp32 weight -> find abs max / 448.
    then x / s for the quantized version then cast to fp8
    """
    # assert weight.shape[0] % block_size == 0 and weight.shape[1] % block_size == 0, f"{key}: {weight.shape}"

    num_blocks_y = math.ceil(weight.shape[0] / block_size)
    num_blocks_x = math.ceil(weight.shape[1] / block_size)
    if weight.shape[0] % block_size != 0 or weight.shape[1] % block_size != 0:
        print(f"Warning! {key}: {weight.shape}, will create a scale that is shape: {num_blocks_y, num_blocks_x}")
    weight_quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty((num_blocks_y, num_blocks_x), dtype=torch.float32)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            quantized_block, scale = compute_single_block_scaling_and_weights(
                weight=weight[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
            )
            weight_quantized[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = quantized_block
            scales[i, j] = scale
    return weight_quantized, scales

def compute_single_block_scaling_and_weights(weight):
    # this is expected to be sliced alr
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = weight.float().abs().max() / 448.0
    weight = weight / scale
    quantized = weight.to(torch.float8_e4m3fn)

    return quantized, scale

import functools
import math
from typing import Optional

import torch
import torch.nn as nn

from .layers import RMSNorm
from .utils import apply_rotary_emb, sync_grad
from ..flash_attention_utils import hf_compatible_fa2_forward


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer - Fixed for DTensor TP
    """

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.args = args
        self.register_buffer("k_cache", None)
        self.register_buffer("v_cache", None)
        # Calculate padding needed for v
        self.v_padding_size = self.qk_head_dim - self.v_head_dim
        # Initialize a zero padding tensor that we'll reuse
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if not args.no_kv_cache:
            self._init_kv_cache(args)

        self.tp_mesh = args.tp_mesh

    def _init_kv_cache(self, args):
        """Initialize KV cache buffers"""
        self.register_buffer(
            "k_cache",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), dtype=torch.bfloat16),
            persistent=False,
        )

    def adjust_for_tp(self, tp_size):
        """Adjust attention for tensor parallelism"""
        self.n_local_heads = self.n_heads // tp_size

        # Resize the cache tensors
        self.k_cache = self.k_cache[:, :, : self.n_local_heads, :]
        self.v_cache = self.v_cache[:, :, : self.n_local_heads, :]

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """
        Forward pass for MLA - Fixed for DTensor TP shape handling
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if self.args.use_tp and k_pe.requires_grad:
            k_pe.retain_grad()
            k_pe.register_hook(functools.partial(sync_grad, name="k_pe", group=self.args.tp_mesh.get_group("tp")))

        kv = self.wkv_b(self.kv_norm(kv))

        # Handle DTensor sharding for kv as well
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)

        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

        # Pad v to match qk_head_dim if necessary
        if self.v_padding_size > 0:
            v_padded = torch.cat(
                [v, torch.zeros(bsz, seqlen, self.n_local_heads, self.v_padding_size, device=v.device, dtype=v.dtype)], dim=-1
            )

        else:
            v_padded = v

        # Handle KV caching
        if self.k_cache is None or self.v_cache is None:
            k_to_use = k
            v_to_use = v_padded
        else:
            # Update cache with new K and V values
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v_padded
            # Use the cached values up to current position
            k_to_use = self.k_cache[:bsz, :end_pos]
            v_to_use = self.v_cache[:bsz, :end_pos]

        attn_out, _ = hf_compatible_fa2_forward(
            query_states=q,
            key_states=k_to_use,
            value_states=v_to_use,
            attention_mask=attention_mask,
            softmax_scale=self.softmax_scale,
            deterministic=True,
        )

        attn_out = attn_out[..., : self.v_head_dim]

        x = self.wo(attn_out.flatten(2))

        return x

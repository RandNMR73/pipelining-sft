import functools
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed as dist
from train_utils import ForCausalLMLoss
from torch import nn
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributions import constraints
from torch.utils.checkpoint import checkpoint

from .attention import MLA
from .layers import MLP, MoE, RMSNorm
from .utils import precompute_freqs_cis


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    no_kv_cache: bool = False

    tp_mesh = None
    use_tp= True


def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.args = args

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleDict): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleDict()
        self.args = args
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = Block(layer_id, args)

        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.gradient_checkpointing = False
        self.set_loss_function()

    def set_loss_function(self):
        def _loss(*args, **kwargs):
            loss = ForCausalLMLoss(*args, **kwargs, vocab_size=self.args.vocab_size)
            return loss
        self.loss_function = _loss

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        start_pos: int = 0,
        logits_to_keep=None,
        is_training=True,
        left_pad_tokens=None, 
        logger=None, 
        **kwargs,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = input_ids.size(1)

        if self.embed is None:
            h = input_ids
        else:
            h = self.embed(input_ids)

        if left_pad_tokens is None:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]  # Shape: [seqlen, freqs_dim]
        else:
            batch_size = len(left_pad_tokens)
            freqs_cis = torch.zeros(batch_size, seqlen, self.freqs_cis.shape[-1], 
                                dtype=self.freqs_cis.dtype, device=self.freqs_cis.device)
            
            for i, n_pad in enumerate(left_pad_tokens):
                n_pad = int(n_pad)
                if n_pad < seqlen:  # Safety check to avoid out-of-bounds
                    # For non-padding positions, use the appropriate freqs_cis values
                    freqs_cis[i, n_pad:] = self.freqs_cis[start_pos : seqlen-n_pad]

        if not is_training:
            # inference mode its just a naive triu mask
            assert attention_mask is None, f"you shouldnt need a custom attention mask for inference.."
            attention_mask = None  # no need for fa
        else:
            # training causal mask
            assert attention_mask is not None, f"you need a custom attention mask for training.."
            # this is super simplified version of mask creation from HF, literally that simple
            # b l
            attention_mask = attention_mask.to(device=h.device, dtype=torch.bool)
            if attention_mask.all():
                attention_mask = None

        for layer in self.layers.values():
            h = gradient_checkpointing(
                module=layer,
                enabled=self.training and self.gradient_checkpointing,
                x=h,
                start_pos=start_pos,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
            )

        if self.norm is not None:
            h = self.norm(h)

        if self.head is None:
            # intermediate stages
            return h

        # here is for last stage only
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if logits_to_keep is None:
            logits = self.head(h)
        else:
            logits = self.head(h[torch.arange(h.size(0)), logits_to_keep])
            logits = logits.unsqueeze(1)
            
        if labels is None:
            # only necessary logits
            return logits

        # here means doing loss function
        assert (
            logits_to_keep == None
        ), f"this should only be non default when doing logit return not loss return, value: {logits_to_keep}"
        # s unsqueeze for torchtitan PP compatibility
        loss = self.loss_function(logits=logits, labels=labels, **kwargs).unsqueeze(0)

        return loss


def pipeline_parallelize_model(model, num_stages, stage_idx):
    assert stage_idx < num_stages, f"Stage {stage_idx} is not in the model"
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"Creating model stage {stage_idx} of {num_stages}")

    model.embed = model.embed if stage_idx == 0 else None
    model.head = model.head if stage_idx == num_stages - 1 else None
    model.norm = model.norm if stage_idx == num_stages - 1 else None

    # remove layers
    all_layers = model.layers
    num_layers = len(all_layers)
    division = num_layers // num_stages
    residual = num_layers % num_stages

    layers_per_stage = [division + 1 if stage < residual else division for stage in range(num_stages)]
    assert sum(layers_per_stage) == num_layers
    layer_id_start = sum(layers_per_stage[:stage_idx])
    layer_id_end = layer_id_start + layers_per_stage[stage_idx]
    for layer_id in range(num_layers):
        if layer_id < layer_id_start or layer_id >= layer_id_end:
            del model.layers[str(layer_id)]


def tensor_parallelize_model(model, tp_mesh, no_tp_shard_on_experts=True, shard_embedding=True, shard_head=True, shard_mlp=True):
    tp_size = tp_mesh.size()
    # 1) Embedding & head
    if model.embed is not None and shard_embedding:
        parallelize_module(
            model,
            tp_mesh,
            {
                "embed": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Replicate(),
                ),
            },
        )

    if model.head is not None and shard_head:
        parallelize_module(
            model,
            tp_mesh,
            {
                "head": ColwiseParallel(
                    output_layouts=Replicate(),  # Ensures full logits
                )
            },
        )

    # 2) Per‐Block TP plan
    for layer_name, blk in model.layers.items():
        # ---- attention proj shards (same as before) ----
        parallelize_module(
            blk,
            tp_mesh,
            {
                "attn.wq": ColwiseParallel(),
                "attn.wq_b": ColwiseParallel(),
                "attn.wkv_b": ColwiseParallel(),
                "attn.wo": RowwiseParallel(),
            },
        )
        blk.attn.n_local_heads = blk.attn.n_local_heads // tp_size

        if blk.attn.k_cache is not None and blk.attn.v_cache is not None:
            blk.attn.adjust_for_tp(tp_size)

        # ---- feed‐forward / MoE shards ----
        if isinstance(blk.ffn, MoE):
            if not no_tp_shard_on_experts:
                for idx, expert in enumerate(blk.ffn.experts):
                    parallelize_module(
                        expert,
                        tp_mesh,
                        {
                            "w1": ColwiseParallel(),
                            "w2": RowwiseParallel(),
                            "w3": ColwiseParallel(),
                        },
                    )

            # shard the shared‐experts MLP
            parallelize_module(
                blk.ffn.shared_experts,
                tp_mesh,
                {
                    "w1": ColwiseParallel(),
                    "w2": RowwiseParallel(),
                    "w3": ColwiseParallel(),
                },
            )

        elif shard_mlp:
            # standard MLP block
            parallelize_module(
                blk.ffn,
                tp_mesh,
                {
                    "w1": ColwiseParallel(),
                    "w2": RowwiseParallel(),
                    "w3": ColwiseParallel(),
                },
            )

    return model

def materialize_meta_module(module: torch.nn.Module, device: torch.device):
    """
    Recursively replace all meta parameters and buffers in the model with torch.empty() or zeros(),
    matching the original shape, dtype, and device — WITHOUT calling reset_parameters().
    """
    for name, param in list(module.named_parameters(recurse=False)):
        # if param.device.type == "meta":
        new_param = torch.empty(param.shape, dtype=param.dtype, device=device)
        module.register_parameter(name, torch.nn.Parameter(new_param))
    for name, buffer in list(module.named_buffers(recurse=False)):
        if buffer.device.type == "meta":
            new_buf = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
            module.register_buffer(name, new_buf)
    for child in module.children():
        materialize_meta_module(child, device)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())

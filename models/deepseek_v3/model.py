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

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has
                old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std
                equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of
                causal language models,
                where the generated tokens' probabilities won't be affected by the added embeddings because initializing
                the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before
                and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Since we are basically reusing the same old embeddings with new weight values, gathering is required
        vocab_size = model_embeds.weight.shape[0]

        # Update base model and current model config.
        self.args.vocab_size = vocab_size
        self.vocab_size = vocab_size
        self.set_loss_function()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing)

        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        old_lm_head = self.get_output_embeddings()
        if isinstance(old_lm_head, torch.nn.Embedding):
            new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
        else:
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)

        old_lm_head_requires_grad = old_lm_head.weight.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution
                that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero
                and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of
                the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added
                embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next
                token probability before and after adding the new embeddings.
                Refer to this article for more information:
                https://nlp.stanford.edu/~johnhew/vocab-expansion.html


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, \
                        which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            print(
                "You are resizing the embedding layer without providing a"
                "`pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance \
                    reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct \
                    value for resizing, refer to this guide:"
            )

        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            raise NotImplementedError(f"We dont support non mean resizing of embeddings...")
            self._init_weights(new_embeddings)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new embeddings  (in particular added tokens). The new embeddings will be initialized
            # from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            added_num_tokens = new_num_tokens - old_num_tokens
            self._init_added_embeddings_weights_with_mean(
                old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
            )

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        # Replace weights in old_embeddings and return to maintain the same embedding type.
        # This ensures correct functionality when a Custom Embedding class is passed as input.
        # The input and output embedding types remain consistent. (c.f. https://github.com/huggingface/transformers/pull/31979)
        old_embeddings.weight.data = new_embeddings.weight.data
        old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
        if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
            old_embeddings.padding_idx = None

        return old_embeddings

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: Optional[int] = None,
        transposed: Optional[bool] = False,
        mean_resizing: bool = True,
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate
                normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that
                has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing
                the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by
                the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the
                next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """

        if new_num_tokens is None:
            return old_lm_head

        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            raise NotImplementedError(f"We dont support non mean resizing of embeddings...")
            self._init_weights(new_lm_head)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new lm_head weights (in particular added tokens). The new lm_head weights
            # will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

            added_num_tokens = new_num_tokens - old_num_tokens
            self._init_added_lm_head_weights_with_mean(
                old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
            )
            if has_new_lm_head_bias:
                self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        self._copy_lm_head_original_to_resized(new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias)

        return new_lm_head

    def _init_added_embeddings_weights_with_mean(
        self, old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
    ):
        old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
        mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
        old_centered_embeddings = old_embeddings_weight - mean_embeddings
        covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

        # Check if the covariance is positive definite.
        epsilon = 1e-9
        is_covariance_psd = constraints.positive_definite.check(epsilon * covariance).all()
        if is_covariance_psd:
            # If covariances is positive definite, a distribution can be created. and we can sample new weights from it.
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embeddings, covariance_matrix=epsilon * covariance
            )
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = distribution.sample(sample_shape=(added_num_tokens,)).to(
                old_embeddings.weight.dtype
            )
        else:
            # Otherwise, just initialize with the mean. because distribution will not be created.
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = (
                mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
            )

    def _init_added_lm_head_weights_with_mean(
        self,
        old_lm_head,
        new_lm_head,
        old_lm_head_dim,
        old_num_tokens,
        added_num_tokens,
        transposed=False,
    ):
        if transposed:
            # Transpose to the desired shape for the function.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

        # The same initialization logic as Embeddings.
        self._init_added_embeddings_weights_with_mean(
            old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens
        )

        if transposed:
            # Transpose again to the correct shape.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

    def _init_added_lm_head_bias_with_mean(self, old_lm_head, new_lm_head, added_num_tokens):
        bias_mean = torch.mean(old_lm_head.bias.data, axis=0, dtype=torch.float32)
        bias_std = torch.std(old_lm_head.bias.data, axis=0).to(torch.float32)
        new_lm_head.bias.data[-1 * added_num_tokens :].normal_(mean=bias_mean, std=1e-9 * bias_std)

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

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

    # time to remove stuff
    model.embed = model.embed if stage_idx == 0 else None
    model.head = model.head if stage_idx == num_stages - 1 else None
    model.norm = model.norm if stage_idx == num_stages - 1 else None

    # remove layers!!
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

import math
from typing import Iterable, Tuple, Optional
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import yaml 

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@dataclass
class Config:
    """Configuration dataclass loaded from YAML"""
    model: Dict[str, Any]
    training: Dict[str, Any]
    dataset: Dict[str, Any]
    distributed: Dict[str, Any]
    checkpoint: Dict[str, Any]
    logging: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)



@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.

        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm

@torch.no_grad()
def clip_grad_norm_no_tp_(
    named_parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    ep_mesh: DeviceMesh | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    this handles the case of PP + EP with no TP
    """
    assert not any([isinstance(p.grad, DTensor) for n, p in named_parameters]), f"this is not used for TPed models"

    named_parameters = [(n, p) for n, p in named_parameters if p.grad is not None]

    expert_grads = [p.grad for n, p in named_parameters if "shared_experts" not in n and "experts" in n]
    non_expert_grads = [p.grad for n, p in named_parameters if "shared_experts" in n or "experts" not in n]

    non_expert_norm = torch.nn.utils.get_total_norm(non_expert_grads, norm_type, error_if_nonfinite, foreach)
    expert_norm = torch.nn.utils.get_total_norm(expert_grads, norm_type, error_if_nonfinite, foreach)
    if len(expert_grads) == 0 or expert_norm == 0.0:
        # in case no experts on this rank was activated so no grads,
        # we need to set to zero tensor on gpu to ensure all reduce across EP doesnt crash
        expert_norm = torch.zeros_like(non_expert_norm)

    if ep_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(expert_norm, op=dist.ReduceOp.MAX, group=ep_mesh.get_group())
        else:
            expert_norm **= norm_type
            dist.all_reduce(expert_norm, op=dist.ReduceOp.SUM, group=ep_mesh.get_group())
            expert_norm **= 1.0 / norm_type

    total_norm = (non_expert_norm**norm_type + expert_norm**norm_type) ** (1.0 / norm_type)

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    parameters = [p for _, p in named_parameters]
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)

    return total_norm


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        if not isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = torch.tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype)
        elif num_items_in_batch.device != loss.device:
            num_items_in_batch = num_items_in_batch.to(loss.device)

        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def clear_cache_and_report(stage_name):
    import gc

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.synchronize()
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"global_rank {rank} on {stage_name}, \
        GPU USAGE REPORT!! reserved: {torch.cuda.memory_reserved() / 1024 **3} GB, \
        allocated: {torch.cuda.memory_allocated() / 1024 **3} GB"
    )

def is_gradient_accumulation_boundary(step=None, gas_amount=None):
    return (step + 1) % gas_amount == 0

def take_optimizer_step(
    model,
    optimizer,
    lr_scheduler=None,
    is_gas_boundary=None,
    grad_clip_max_norm=1.0,
    tp_mesh=None,
    pp_mesh=None,
    config=None,
    fp8_training=False,
):
    if is_gas_boundary:
        for p in model.parameters():
            if p.grad is not None:
                if config.get("gradient_accumulation_steps", 1) > 1:
                    p.grad = p.grad / (config.get("gradient_accumulation_steps", 1) * config.get("num_mbs", 32))

        if fp8_training:
            grad_norm = clip_grad_norm_no_tp_(
                list(model.named_parameters()), max_norm=grad_clip_max_norm, foreach=False, pp_mesh=pp_mesh, ep_mesh=tp_mesh
            )
        else:
            grad_norm = clip_grad_norm_(
                [p for p in model.parameters()], max_norm=grad_clip_max_norm, pp_mesh=pp_mesh, foreach=False
            )

        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        return grad_norm

    else:
        return None

import os
import json
from typing import Dict, Set, Optional, Tuple
from collections import defaultdict
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from torch import nn
import torch.distributed as dist
from pathlib import Path
from glob import glob 

import subprocess

# Import the weight_dequant function from kernel
from .kernel import weight_dequant
from .fp8_layers import per_block_cast_to_fp8

# Import FP8 Linear layer for type checking
try:
    from .fp8_layers import Linear as FP8Linear
    FP8_AVAILABLE = True
except ImportError:
    FP8Linear = None
    FP8_AVAILABLE = False

# Mapping from HuggingFace naming to model naming
HF_TO_MODEL_MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
    "e_score_correction_bias": ("bias", None),  # For gate bias
}

MODEL_TO_HF_MAPPING = {
    # Embeddings and top-level components
    "embed": "embed_tokens",
    "norm": "norm",
    "head": "lm_head",
    
    # Layer normalization
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    
    # Attention components
    "wq": "q_b_proj",      # Note: wq maps to q_b_proj (not q_proj)
    "wq_a": "q_a_proj",
    "wq_b": "q_b_proj",
    "q_norm": "q_a_layernorm",
    "wkv_a": "kv_a_proj_with_mqa",
    "wkv_b": "kv_b_proj",
    "kv_norm": "kv_a_layernorm",
    "wo": "o_proj",
    
    # FFN components
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
    "gate": "gate",
    
    # Other components
    "scale": "scale",
    "bias": "e_score_correction_bias",
}

# Column-parallel layers (split on output dimension)
COLUMN_PARALLEL_COMPONENTS = {'wq', 'wq_b', 'wkv_b', 'w1', 'w3', 'head', 'embed'}

# Row-parallel layers (split on input dimension)  
ROW_PARALLEL_COMPONENTS = {'wo', 'w2'}

# Non-TP layers (layer norms, embeddings, etc.)
NON_TP_COMPONENTS = {'attn_norm', 'ffn_norm', 'q_norm', 'kv_norm', 'embed', 'norm', 'head', 'gate', 'wq_a', 'wkv_a'}


def is_fp8_linear_parameter(param_path: str, model: torch.nn.Module) -> bool:
    """
    Check if a parameter belongs to an FP8 Linear layer.
    
    Args:
        param_path: Dot-separated path to the parameter (e.g., "layers.0.ffn.w1.weight")
        model: The model to check
        
    Returns:
        bool: True if the parameter belongs to an FP8 Linear layer
    """
    if not FP8_AVAILABLE:
        return False
    
    try:
        # Navigate to the parent module (not the parameter itself)
        parts = param_path.split(".")
        if parts[-1] in ["weight", "bias"]:
            module_path = ".".join(parts[:-1])  # Remove "weight" or "bias"
        else:
            module_path = param_path
        
        # Navigate to the module
        module = model
        for part in module_path.split("."):
            if part == "layers":
                continue  # Skip "layers" as it's handled specially
            if part.isdigit():
                # Handle layer indices
                if hasattr(module, "layers"):
                    if isinstance(module.layers, nn.ModuleDict):
                        module = module.layers[part]
                    else:
                        module = module.layers[int(part)]
                else:
                    return False
            elif part == "experts":
                continue  # Skip "experts" as the next part will be the index
            elif part.isdigit() and hasattr(module, "experts"):
                # Handle expert indices
                expert_idx = int(part)
                if expert_idx < len(module.experts):
                    module = module.experts[expert_idx]
                else:
                    return False
            elif hasattr(module, part):
                module = getattr(module, part)
            else:
                return False
        
        # Check if the final module is an FP8Linear
        return isinstance(module, FP8Linear)
        
    except (AttributeError, IndexError, ValueError):
        return False


def should_load_in_fp32(param_path: str, model: torch.nn.Module, use_fp8_training: bool = False) -> bool:
    """
    Determine if a parameter should be loaded in FP32 precision.
    
    Args:
        param_path: Dot-separated path to the parameter
        model: The model to check
        use_fp8_training: Whether FP8 training is enabled
        
    Returns:
        bool: True if the parameter should be loaded in FP32
    """
    if not use_fp8_training:
        return False
        
    # Always load FP8 Linear layer weights in FP32 for accurate computation
    return is_fp8_linear_parameter(param_path, model)


def convert_hf_name_to_model_name(hf_name: str) -> str:
    """Convert HuggingFace weight name to model weight name."""
    if hf_name.startswith("model."):
        hf_name = hf_name[len("model.") :]

    # Skip layer 61 as in the original script
    if "layers.61" in hf_name:
        return None

    # Replace naming conventions
    model_name = hf_name.replace("self_attn", "attn")
    model_name = model_name.replace("mlp", "ffn")
    model_name = model_name.replace("weight_scale_inv", "scale")
    model_name = model_name.replace("e_score_correction_bias", "bias")

    # Extract the key (second to last component)
    parts = model_name.split(".")
    if len(parts) < 2:
        return None

    key = parts[-2]
    if key not in HF_TO_MODEL_MAPPING:
        return None

    new_key, _ = HF_TO_MODEL_MAPPING[key]
    model_name = model_name.replace(key, new_key)

    return model_name


def get_layer_assignment(num_layers: int, pp_size: int, pp_rank: int) -> tuple:
    """
    Calculate which layers belong to a specific pipeline stage.

    Returns:
        tuple: (layer_start, layer_end) indices for this PP stage
    """
    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size

    # Calculate layer distribution
    layer_start = sum(layers_per_stage + (1 if i < remainder else 0) for i in range(pp_rank))
    layer_end = layer_start + layers_per_stage + (1 if pp_rank < remainder else 0)

    return layer_start, layer_end


def load_pipeline_weights(
    model: torch.nn.Module,
    model_dir: str,
    pp_rank: int,
    pp_size: int,
    device: torch.device,
    n_layers: int = 61,  # From ModelArgs
    block_size: int = 128,
    use_fp8_training: bool = False
) -> int:
    """
    Load weights for a specific pipeline parallel stage of the model.

    Args:
        model: The initialized model (on meta device) with PP sharding applied
        hf_ckpt_path: Path to the directory containing HuggingFace checkpoint files
        pp_rank: Pipeline parallel rank of current process
        pp_size: Total number of pipeline parallel stages
        device: Device to load weights to
        n_layers: Total number of transformer layers in the model
        block_size: Block size for FP8 dequantization
        use_fp8_training: Whether to use FP8 training mode - if True, will load weights in FP32 if they are FP8 Linear layer weights

    Returns:
        int: Total number of parameters loaded
    """
    print(f"Loading weights for PP stage {pp_rank}/{pp_size}")

    # Load model index

    model_index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(model_index_path, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Determine which layers belong to this PP stage
    layer_start, layer_end = get_layer_assignment(n_layers, pp_size, pp_rank)
    stage_layers = set(range(layer_start, layer_end)) if layer_start < layer_end else set()
    print(f"PP stage {pp_rank}: responsible for layers {layer_start} to {layer_end-1}")

    # Group weights by file to minimize file loads
    weights_by_file = defaultdict(list)
    scale_weights_needed = {}  # Track scale weights needed for FP8 conversion

    # First pass: identify which weights we need for this PP stage
    for hf_name, file_name in weight_map.items():
        model_name = convert_hf_name_to_model_name(hf_name)
        if model_name is None:
            continue

        # Check if this weight belongs to this PP stage
        should_load = False

        # Check embedding (only first stage)
        if model_name == "embed.weight":
            should_load = pp_rank == 0

        # Check output layers (only last stage)
        elif model_name in ["norm.weight", "head.weight", "head.bias"]:
            should_load = pp_rank == pp_size - 1

        # Check transformer layers
        elif "layers." in model_name:
            # Extract layer index
            parts = model_name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        should_load = layer_idx in stage_layers
                        break
                    except ValueError:
                        continue

        if not should_load:
            continue

        # For scale weights, skip them in first pass (they'll be loaded with their weights)
        if hf_name.endswith("_scale_inv") or hf_name.endswith(".scale"):
            continue

        # Track which weights might need scale factors
        if "weight" in hf_name and not hf_name.endswith("_scale_inv"):
            scale_name = hf_name.replace(".weight", ".weight_scale_inv")
            if scale_name in weight_map:
                scale_weights_needed[hf_name] = scale_name

        weights_by_file[file_name].append((hf_name, model_name))

    # Load weights
    loaded_files_cache = {}
    total_loaded = 0

    def get_tensor_from_file(file_name: str, tensor_name: str) -> torch.Tensor:
        """Load a tensor from a safetensor file with caching."""
        if file_name not in loaded_files_cache:
            file_path = os.path.join(model_dir, file_name)
            # Load to CPU first to manage memory
            loaded_files_cache[file_name] = load_file(file_path, device="cpu")
        return loaded_files_cache[file_name][tensor_name]

    # Process each file
    print(f"PP stage {pp_rank}: Processing {len(weights_by_file)} files...")
    for file_name, weight_list in tqdm(weights_by_file.items(), desc=f"Loading weights for PP stage {pp_rank}"):
        for hf_name, model_name in weight_list:
            # Get the tensor
            tensor = get_tensor_from_file(file_name, hf_name)

            # Handle FP8 conversion if needed
            if tensor.element_size() == 1:  # FP8 weight
                scale_hf_name = scale_weights_needed.get(hf_name)
                if scale_hf_name:
                    scale_file = weight_map[scale_hf_name]
                    scale_tensor = get_tensor_from_file(scale_file, scale_hf_name)
                    # Move tensors to GPU for dequantization
                    tensor = tensor.to(device)
                    scale_tensor = scale_tensor.to(device)
                    # Ensure tensors are contiguous
                    tensor = tensor.contiguous()
                    scale_tensor = scale_tensor.contiguous()
                    
                    # Dequantize - use FP32 if this is an FP8 Linear layer weight
                    if use_fp8_training:
                        tensor = weight_dequant(tensor, scale_tensor, block_size=block_size, dtype=torch.float32)
                    else:
                        tensor = weight_dequant(tensor, scale_tensor, block_size=block_size)
                else:
                    if use_fp8_training:
                        tensor = tensor.to(device, dtype=torch.float32)
                    else:
                        tensor = tensor.to(device)
            else:
                # For non-FP8 tensors, check if they should be loaded in FP32
                if use_fp8_training:
                    tensor = tensor.to(device, dtype=torch.float32)
                    print(f"Loading {model_name} in FP32 for FP8 training (standard tensor)")
                else:
                    tensor = tensor.to(device)

            # Assign to model
            try:
                # Navigate to the correct parameter in the model
                param = model
                parts = model_name.split(".")

                # Handle different model structures
                i = 0
                while i < len(parts):
                    if parts[i] == "layers" and i + 1 < len(parts):
                        # For pipeline model, layers are in layers_dict
                        layer_idx = parts[i + 1]
                        if hasattr(param, "layers_dict"):
                            if layer_idx in param.layers_dict:
                                param = param.layers_dict[layer_idx]
                                i += 2
                            else:
                                raise AttributeError(f"Layer {layer_idx} not in layers_dict")
                        else:
                            if isinstance(param.layers, nn.ModuleDict):
                                # Fallback to regular layers
                                param = param.layers[str(layer_idx)]
                                i += 2
                            else:
                                # Fallback to regular layers
                                param = param.layers[int(layer_idx)]
                                i += 2
                    elif parts[i] == "experts" and i + 1 < len(parts):
                        # Get the expert
                        expert_idx = int(parts[i + 1])
                        param = param.experts[expert_idx]
                        if param is None: # for EP layers where experts are not defined
                            break
                        i += 2
                    elif hasattr(param, parts[i]):
                        param = getattr(param, parts[i])
                        i += 1
                    else:
                        raise AttributeError(f"Cannot find attribute {parts[i]} in {type(param).__name__}")

                # Assign the tensor
                if isinstance(param, torch.nn.Parameter):
                    if param.shape != tensor.shape:
                        print(f"Shape mismatch for {model_name}: param shape={param.shape}, tensor shape={tensor.shape}")
                        raise ValueError(f"Shape mismatch")
                    param.data.copy_(tensor)
                    total_loaded += tensor.numel()
                else:
                    # For other attributes
                    if param is None:
                        continue
                    elif hasattr(param, "data"):
                        param.data.copy_(tensor)
                        total_loaded += tensor.numel()
                    else:
                        raise ValueError(f"Cannot assign tensor to {model_name}")

            except Exception as e:
                print(f"Warning: Failed to assign weight {model_name}: {e}.")
                continue

        # Clear cache periodically to manage memory
        if len(loaded_files_cache) > 3:
            oldest_file = next(iter(loaded_files_cache))
            del loaded_files_cache[oldest_file]
            torch.cuda.empty_cache()

    print(f"PP stage {pp_rank}: Weight loading complete. Loaded {total_loaded:,} parameters")
    return total_loaded


def create_partial_model_index(
    filepath: str,
    pp_rank: int,
    tp_rank: int,
    pp_size: int,
    tp_size: int,
    is_main_file: bool = True, 
    temporal_dir: Optional[str] = None
) -> dict:
    """
    Create a partial index for a single checkpoint file.
    
    Args:
        filepath: Path to the safetensors file
        pp_rank: Pipeline parallel rank
        tp_rank: Tensor parallel rank
        pp_size: Total pipeline parallel size
        tp_size: Total tensor parallel size
        is_main_file: Whether this is the main checkpoint file (vs expert-only file)
    
    Returns:
        Dictionary containing partial index information
    """
    # Determine the filename in the final checkpoint
    if is_main_file:
        filename = f"model-stage-{pp_rank:03d}-of-{pp_size:03d}.safetensors"
    else:
        filename = f"model-stage-{pp_rank:03d}-of-{pp_size:03d}-experts-tp{tp_rank:02d}.safetensors"
    
    weight_map = {}
    file_size = 0
    
    if os.path.exists(filepath):
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Read the safetensors file to get weight names
        try:
            with safe_open(filepath, framework="pt") as f:
                # Get all tensor names in this file
                for tensor_name in f.keys():
                    weight_map[tensor_name] = filename
        except Exception as e:
            print(f"[PP{pp_rank}/TP{tp_rank}] Warning: Could not read {filepath}: {e}")
    
    # Create partial index structure
    partial_index = {
        "pp_rank": pp_rank,
        "tp_rank": tp_rank,
        "filename": filename,
        "file_size": file_size,
        "weight_map": weight_map,
        "is_main_file": is_main_file
    }
    
    # Create filename for partial index
    if is_main_file:
        partial_index_filename = f"partial_index_pp{pp_rank:03d}_tp{tp_rank:03d}.json"
    else:
        partial_index_filename = f"partial_index_pp{pp_rank:03d}_tp{tp_rank:03d}_experts.json"
        
    local_path = os.path.join(temporal_dir, partial_index_filename)

    with open(local_path, "w") as f:
        json.dump(partial_index, f, indent=2)
    
    return local_path

def upload_and_delete(cmd_to_upload, local_path, output_dir):

    cmd = cmd_to_upload.format(local_path=local_path, output_dir=os.path.join(output_dir, os.path.basename(local_path)))
    subprocess.run(cmd, shell=True, check=True)
    os.remove(local_path)

    return os.path.join(output_dir, os.path.basename(local_path))


def export_tp_pp_stage_to_hf_format_with_experts(
    model: torch.nn.Module,
    pp_rank: int,
    pp_size: int,
    tp_rank: int,
    tp_size: int,
    tp_mesh,
    temporal_dir: str = "./checkpoints",
    iteration: Optional[int] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16, 
    using_expert_parallel: bool = True,
    output_dir: Optional[str] = None,
    use_fp8_quantization=False,
    fp8_mp=False,

) -> Optional[str]:
    """
    Export a tensor+pipeline parallelized model stage to HuggingFace format.
    
    When using_expert_parallel=True:
    - Rank 0: Saves everything (all non-expert weights + its own experts) in one file
    - Ranks 1-7: Save only their experts in separate files
    
    Example with tp_size=8:
    - Rank 0: Saves all non-expert weights + experts 0-31
    - Rank 1: Saves only experts 32-63
    - Rank 2: Saves only experts 64-95
    - ... and so on
    
    When using_expert_parallel=False:
    - Only tp_rank 0 saves everything (original behavior)
    
    Args:
        model: The TP+PP model for this stage
        pp_rank: Pipeline stage rank
        pp_size: Total number of pipeline stages
        tp_rank: Tensor parallel rank
        tp_size: Total number of TP ranks
        tp_mesh: The tensor parallel device mesh
        temporal_dir: Directory to save checkpoints temporarily befor uploading to output_dir 
        iteration: Optional iteration number for checkpoint naming
        dtype: Optional dtype to cast parameters to before saving
        using_expert_parallel: if True, each rank saves its own experts
        use_fp8_quantization: if True, we will apply fp8 quantization to the weights
            and fp32 scale will be stored in parameter and index file
        fp8_mp: If True, we are using FP8 training and thus no TP 
    Returns:
        Path to saved file
    """
    assert using_expert_parallel, f"using_expert_parallel being False not supported!"

    def random_sleep(start=10, end=60):
        """Random sleep to simulate distributed training environment."""
        import random
        import time
        time.sleep(random.uniform(start, end))

    if fp8_mp:
        assert use_fp8_quantization

    if use_fp8_quantization:
        assert all(
            [p.dtype == torch.float32 or hasattr(p, "fp_param") for p in model.parameters()]
        ), f"you want to fp8 quantize but you don't have fp params... RED FLAG"

    # Prepare state dicts
    state_dict = {}  # For non-expert weights (rank 0 only)
    expert_state_dict = {}  # For expert weights (each rank saves its own)
    if output_dir is not None and output_dir.startswith("gs://"):
        upload_to_gcs = True 
        cmd_to_upload = 'gsutil cp -r {local_path} {output_dir}'
    else:
        os.makedirs(output_dir, exist_ok=True)
        upload_to_gcs = False
        cmd_to_upload = 'cp -r {local_path} {output_dir}'
    
    # Helper function to process a parameter
    def process_param(
        param: torch.nn.Parameter, model_name: str, hf_key: str, component_name: str, use_fp8_quantization=False
    ):
        if fp8_mp:
            is_tp, is_column_parallel = False, False
        else:
            is_tp, is_column_parallel = is_tensor_parallel_layer(model_name, component_name)
        
        if is_tp and tp_size > 1:
            # Gather TP weights on rank 0
            gathered_param = gather_tp_weights(param, tp_mesh, tp_rank, tp_size, is_column_parallel)
            if tp_rank == 0 and gathered_param is not None:
                state_dict[hf_key] = gathered_param
                if use_fp8_quantization:
                    quantized_weight, scale = per_block_cast_to_fp8(state_dict[hf_key])
                    state_dict[hf_key] = quantized_weight
                    state_dict[hf_key.replace(".weight", ".weight_scale_inv")] = scale
                else:
                    if model_name != "bias" and dtype is not None:
                        state_dict[hf_key] = state_dict[hf_key].to(dtype)
        else:
            # Non-TP layer or TP size is 1, just save directly on rank 0
            if tp_rank == 0:
                # Handle DTensor case for non-TP layers
                if hasattr(param, '_local_tensor'):
                    tensor_data = param._local_tensor
                else:
                    tensor_data = param.data
                    
                state_dict[hf_key] = tensor_data
                if use_fp8_quantization:
                    quantized_weight, scale = per_block_cast_to_fp8(state_dict[hf_key])
                    state_dict[hf_key] = quantized_weight
                    state_dict[hf_key.replace(".weight", ".weight_scale_inv")] = scale
                else:
                    if model_name != "bias" and dtype is not None:
                        state_dict[hf_key] = state_dict[hf_key].to(dtype)
    
    # Process embeddings (only on first PP stage)
    if hasattr(model, 'embed') and model.embed is not None:
        hf_name = MODEL_TO_HF_MAPPING['embed']
        for name, param in model.embed.named_parameters():
            hf_key = f"model.{hf_name}.{name}"
            process_param(param, 'embed', hf_key, 'embed')
        if tp_rank == 0:
            print(f"[Stage {pp_rank}] Found embedding layer")
    
    # Process transformer layers
    if hasattr(model, 'layers') and model.layers is not None:
        layer_ids = sorted(model.layers.keys(), key=lambda x: int(x))
        if tp_rank == 0:
            print(f"[Stage {pp_rank}] Processing layers: {layer_ids}")
        
        for layer_id in layer_ids:
            # Skip layer 61
            if int(layer_id) == 61:
                continue
                
            layer = model.layers[layer_id]
            
            # Attention components
            if hasattr(layer, 'attn'):
                attn = layer.attn
                attn_components = ['wq_a', 'wq_b', 'wq', 'q_norm', 'wkv_a', 'wkv_b', 'kv_norm', 'wo']
                
                for model_name in attn_components:
                    if hasattr(attn, model_name) and model_name in MODEL_TO_HF_MAPPING:
                        component = getattr(attn, model_name)
                        hf_name = MODEL_TO_HF_MAPPING[model_name]
                        for param_name, param in component.named_parameters():
                            hf_key = f"model.layers.{layer_id}.self_attn.{hf_name}.{param_name}"
                            process_param(
                                param,
                                model_name,
                                hf_key,
                                model_name,
                                use_fp8_quantization=use_fp8_quantization and 'norm' not in model_name,
                            )

            # Layer norms
            if hasattr(layer, 'attn_norm'):
                hf_name = MODEL_TO_HF_MAPPING['attn_norm']
                for param_name, param in layer.attn_norm.named_parameters():
                    hf_key = f"model.layers.{layer_id}.{hf_name}.{param_name}"
                    process_param(param, 'attn_norm', hf_key, 'attn_norm')
                        
            if hasattr(layer, 'ffn_norm'):
                hf_name = MODEL_TO_HF_MAPPING['ffn_norm']
                for param_name, param in layer.ffn_norm.named_parameters():
                    hf_key = f"model.layers.{layer_id}.{hf_name}.{param_name}"
                    process_param(param, 'ffn_norm', hf_key, 'ffn_norm')
            
            # FFN components
            if hasattr(layer, 'ffn'):
                ffn = layer.ffn
                
                # Handle individual experts if they exist
                if hasattr(ffn, 'experts'):
                    # Calculate expert distribution
                    n_routed_experts = len(ffn.experts)
                    n_local_experts = n_routed_experts // tp_size
                    experts_start_idx = tp_rank * n_local_experts
                    experts_end_idx = experts_start_idx + n_local_experts
                    
                    print(f"[PP{pp_rank}/TP{tp_rank}] Processing local experts {experts_start_idx}-{experts_end_idx-1} for layer {layer_id}")
                    
                    # Process only this rank's experts
                    for expert_idx in range(experts_start_idx, experts_end_idx):
                        expert = ffn.experts[expert_idx]
                        if expert is None:
                            continue
                        
                        ffn_components = ['w1', 'w2', 'w3']
                        for model_name in ffn_components:
                            if hasattr(expert, model_name) and model_name in MODEL_TO_HF_MAPPING:
                                component = getattr(expert, model_name)
                                hf_name = MODEL_TO_HF_MAPPING[model_name]
                                for param_name, param in component.named_parameters():
                                    hf_key = f"model.layers.{layer_id}.mlp.experts.{expert_idx}.{hf_name}.{param_name}"
                                    
                                    # Get tensor data
                                    if hasattr(param, '_local_tensor'):
                                        tensor_data = param._local_tensor
                                    else:
                                        tensor_data = param.data
                                    if not use_fp8_quantization and dtype is not None:
                                        tensor_data = tensor_data.to(dtype)
                        
                                    if tp_rank == 0:
                                        # Rank 0 saves experts with other weights
                                        state_dict[hf_key] = tensor_data
                                        if use_fp8_quantization:
                                            quantized_weight, scale = per_block_cast_to_fp8(state_dict[hf_key])
                                            state_dict[hf_key] = quantized_weight
                                            state_dict[hf_key.replace(".weight", ".weight_scale_inv")] = scale
                                    else:
                                        # Ranks 1-7 save experts separately
                                        expert_state_dict[hf_key] = tensor_data
                                        if use_fp8_quantization:
                                            quantized_weight, scale = per_block_cast_to_fp8(expert_state_dict[hf_key])
                                            expert_state_dict[hf_key] = quantized_weight
                                            expert_state_dict[hf_key.replace(".weight", ".weight_scale_inv")] = scale
                # Handle shared experts if they exist
                if hasattr(ffn, 'shared_experts'):
                    shared_experts = ffn.shared_experts
                    if tp_rank == 0:
                        print(f"[Stage {pp_rank}] Found shared_experts in layer {layer_id}")
                    
                    # Shared experts have the same structure as regular FFN
                    shared_expert_components = ['w1', 'w2', 'w3']
                    
                    for model_name in shared_expert_components:
                        if hasattr(shared_experts, model_name) and model_name in MODEL_TO_HF_MAPPING:
                            component = getattr(shared_experts, model_name)
                            hf_name = MODEL_TO_HF_MAPPING[model_name]
                            for param_name, param in component.named_parameters():
                                hf_key = f"model.layers.{layer_id}.mlp.shared_experts.{hf_name}.{param_name}"
                                process_param(param, model_name, hf_key, model_name, use_fp8_quantization=use_fp8_quantization)
                
                # Handle the gate (for MoE layers with either experts or shared_experts)
                if hasattr(ffn, 'gate') and 'gate' in MODEL_TO_HF_MAPPING:
                    gate = ffn.gate
                    hf_gate_name = MODEL_TO_HF_MAPPING['gate']
                    
                    # Process all gate parameters
                    for param_name, param in gate.named_parameters():
                        if param_name == 'weight':
                            # Gate weight keeps its name
                            hf_key = f"model.layers.{layer_id}.mlp.{hf_gate_name}.weight"
                            process_param(param, 'gate', hf_key, 'gate')
                        elif param_name == 'bias':
                            # Gate bias needs to be renamed to e_score_correction_bias
                            hf_key = f"model.layers.{layer_id}.mlp.{hf_gate_name}.e_score_correction_bias"
                            process_param(param, 'bias', hf_key, 'bias')
                            if tp_rank == 0:
                                print(f"[Stage {pp_rank}] Found gate bias in layer {layer_id}, mapping to e_score_correction_bias")
                
                # Handle regular FFN (non-MoE) layers - only if no experts or shared_experts
                if not hasattr(ffn, 'experts') and not hasattr(ffn, 'shared_experts'):
                    # Regular FFN layer
                    ffn_components = ['w1', 'w2', 'w3']
                    
                    for model_name in ffn_components:
                        if hasattr(ffn, model_name) and model_name in MODEL_TO_HF_MAPPING:
                            component = getattr(ffn, model_name)
                            hf_name = MODEL_TO_HF_MAPPING[model_name]
                            for param_name, param in component.named_parameters():
                                hf_key = f"model.layers.{layer_id}.mlp.{hf_name}.{param_name}"
                                process_param(param, model_name, hf_key, model_name, use_fp8_quantization=use_fp8_quantization)
    
    # Process final normalization layer (only on last PP stage)
    if hasattr(model, 'norm') and model.norm is not None:
        for param_name, param in model.norm.named_parameters():
            hf_key = f"model.{MODEL_TO_HF_MAPPING['norm']}.{param_name}"
            process_param(param, 'norm', hf_key, 'norm')
        if tp_rank == 0:
            print(f"[Stage {pp_rank}] Found final normalization layer")
    
    # Process output head (only on last PP stage)
    if hasattr(model, 'head') and model.head is not None:
        hf_name = MODEL_TO_HF_MAPPING['head']
        for param_name, param in model.head.named_parameters():
            hf_key = f"{hf_name}.{param_name}"
            process_param(param, 'head', hf_key, 'head')
        if tp_rank == 0:
            print(f"[Stage {pp_rank}] Found language model head")
    
    # Create output directory
    if iteration is not None:
        temporal_dir = os.path.join(temporal_dir, f"iter_{iteration:06d}")

    os.makedirs(temporal_dir, exist_ok=True)
    
    saved_files = []
    
    # Save expert weights for ranks 1-7 only
    if tp_rank > 0 and expert_state_dict:
        # Calculate stats for experts
        expert_params = sum(p.numel() for p in expert_state_dict.values())
        expert_size_gb = sum(p.numel() * p.element_size() for p in expert_state_dict.values()) / (1024**3)
        
        print(f"[PP{pp_rank}/TP{tp_rank}] Expert parameters: {expert_params:,}")
        print(f"[PP{pp_rank}/TP{tp_rank}] Expert size: {expert_size_gb:.2f} GB")
        print(f"[PP{pp_rank}/TP{tp_rank}] Expert weights: {len(expert_state_dict)}")
        
        # Save expert checkpoint
        filename = f"model-stage-{pp_rank:03d}-of-{pp_size:03d}-experts-tp{tp_rank:02d}.safetensors"
        filepath = os.path.join(temporal_dir, filename)
        
        # Save with metadata
        metadata = {
            "format": "pt",
            "pipeline_stage": str(pp_rank),
            "total_stages": str(pp_size),
            "tensor_parallel_rank": str(tp_rank),
            "tensor_parallel_size": str(tp_size),
            "expert_params": str(expert_params),
            "expert_size_gb": f"{expert_size_gb:.2f}",
            "weight_type": "experts",
            "naming_format": "huggingface",
            "iteration": str(iteration) if iteration is not None else "N/A"
        }
        
        save_file(expert_state_dict, filepath, metadata=metadata)
        max_retries = 40  # or float('inf') for infinite
        retry_count = 0

        # Similarly, after saving expert-only checkpoint file (if tp_rank > 0):
        if tp_rank > 0 and os.path.exists(filepath):
            # Create and upload partial index for expert file
            partial_index_file = create_partial_model_index(
                filepath=filepath,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                pp_size=pp_size,
                tp_size=tp_size,
                is_main_file=False, 
                temporal_dir=temporal_dir
            )

        while retry_count < max_retries:
            try:
                random_sleep(30, 60)
                try:                    
                    upload_and_delete(cmd_to_upload, filepath, output_dir)

                    upload_and_delete(cmd_to_upload, partial_index_file, output_dir)

                    print(f"[PP{pp_rank}/TP{tp_rank}] Uploaded checkpoint to path: {os.path.join(output_dir, os.path.basename(filepath))} upon retry {retry_count + 1}")
                    saved_files.append(os.path.join(output_dir, os.path.basename(filepath)))

                except Exception as e:
                    print(f"[PP{pp_rank}/TP{tp_rank}] Failed to upload expert checkpoint to path (attempt {retry_count + 1}): {e}")
                    random_sleep()
                    retry_count += 1
                    continue
        
                break  # Success
                
            except Exception as e:
                print(f"[PP{pp_rank}/TP{tp_rank}] Exception encountered (attempt {retry_count + 1}): {e}")
                random_sleep()
                retry_count += 1
                continue
            
        # Clean up expert memory
        del expert_state_dict
        torch.cuda.empty_cache()
    
    # Save all weights for rank 0 (includes both non-expert weights and rank 0's experts)
    if tp_rank == 0 and state_dict:
        # Calculate stats
        total_params = sum(p.numel() for p in state_dict.values())
        total_size_gb = sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024**3)
        
        n_experts_per_rank = 256 // tp_size
        print(f"[PP{pp_rank}/TP{tp_rank}] Total parameters (including non-expert weights + experts 0-{n_experts_per_rank - 1}): {total_params:,}")

        print(f"[PP{pp_rank}/TP{tp_rank}] Total size: {total_size_gb:.2f} GB")
        print(f"[PP{pp_rank}/TP{tp_rank}] Total weights: {len(state_dict)}")
        
        # Save checkpoint with all weights
        filename = f"model-stage-{pp_rank:03d}-of-{pp_size:03d}.safetensors"
        filepath = os.path.join(temporal_dir, filename)
        
        # Save with metadata
        metadata = {
            "format": "pt",
            "pipeline_stage": str(pp_rank),
            "total_stages": str(pp_size),
            "tensor_parallel_size": str(tp_size),
            "total_params": str(total_params),
            "total_size_gb": f"{total_size_gb:.2f}",
            "naming_format": "huggingface",
            "iteration": str(iteration) if iteration is not None else "N/A",
        }
    
        save_file(state_dict, filepath, metadata=metadata)
        max_retries = 40  # or float('inf') for infinite
        retry_count = 0

        # After saving the main checkpoint file and uploading to output_dir:
        if os.path.exists(filepath):
            # Create and upload partial index for main file
            partial_index_file = create_partial_model_index(
                filepath=filepath,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                pp_size=pp_size,
                tp_size=tp_size,
                is_main_file=True, 
                temporal_dir=temporal_dir
            )

        while retry_count < max_retries:
            try:
                random_sleep(30, 60)
                try:

                    upload_and_delete(cmd_to_upload, filepath, output_dir)

                    upload_and_delete(cmd_to_upload, partial_index_file, output_dir)

                    print(f"[PP{pp_rank}/TP{tp_rank}] Uploaded checkpoint to path: {os.path.join(output_dir, os.path.basename(filepath))} upon retry {retry_count + 1}")
                    saved_files.append(os.path.join(output_dir, os.path.basename(filepath)))

                except Exception as e:
                    print(f"[PP{pp_rank}/TP{tp_rank}] Failed to upload expert checkpoint to path (attempt {retry_count + 1}): {e}")
                    random_sleep()
                    retry_count += 1
                    continue

                break  # Success
                
            except Exception as e:
                print(f"[PP{pp_rank}/TP{tp_rank}] Exception encountered (attempt {retry_count + 1}): {e}")
                random_sleep()
                retry_count += 1
                continue

        if retry_count >= max_retries:
            print(f"[PP{pp_rank}/TP{tp_rank}] Max retries ({max_retries}) exceeded")

        # Clean up memory
        del state_dict
        torch.cuda.empty_cache()
    
    # Return the first saved file path
    return output_dir


def gather_tp_weights(
    param: torch.nn.Parameter,
    tp_mesh: 'DeviceMesh',
    tp_rank: int,
    tp_size: int,
    is_column_parallel: bool = True
) -> torch.Tensor:
    """
    Gather tensor-parallel weights from all TP ranks.
    
    Args:
        param: The local parameter shard (may be a DTensor)
        tp_mesh: The tensor parallel device mesh
        tp_rank: Current TP rank
        tp_size: Total TP size
        is_column_parallel: Whether this is column-parallel (concat on dim 0) or row-parallel (concat on dim 1)
    
    Returns:
        The gathered full weight tensor (only valid on tp_rank 0)
    """
    # Get the local tensor from DTensor if necessary
    if hasattr(param, '_local_tensor'):
        # This is a DTensor, get the local shard
        local_tensor = param._local_tensor
    else:
        # Regular tensor
        local_tensor = param.data
    
    # Determine gather dimension based on parallelism type
    gather_dim = 0 if is_column_parallel else 1
    
    # All ranks need to allocate buffers for all_gather
    gathered_list = [torch.empty_like(local_tensor) for _ in range(tp_size)]
    
    # All-gather the weights
    dist.all_gather(
        gathered_list,
        local_tensor,
        group=tp_mesh.get_group()
    )
    
    # Only rank 0 concatenates and returns the result
    if tp_rank == 0:
        result = torch.cat(gathered_list, dim=gather_dim)
        # Clean up intermediate tensors
        del gathered_list
        return result
    else:
        # Other ranks clean up immediately
        del gathered_list
        torch.cuda.empty_cache()
        return None
    

def is_tensor_parallel_layer(module_name: str, component_name: str) -> Tuple[bool, bool]:
    """
    Determine if a layer is tensor-parallel and its parallelism type.
    
    Returns:
        (is_tp, is_column_parallel): Whether the layer is TP and if it's column-parallel
    """

    if component_name in COLUMN_PARALLEL_COMPONENTS:
        return True, True
    elif component_name in ROW_PARALLEL_COMPONENTS:
        return True, False
    else:
        return False, False
    
def create_model_index_from_files(
    output_dir: str,
    pp_size: int,
    tp_size: int = 1,
    using_expert_parallel: bool = False
) -> None:
    """
    Create the model.safetensors.index.json file by reading saved checkpoint files.
    This includes both the main checkpoint files and expert-only files from ranks 1-7.
    
    Args:
        output_dir: Directory containing the checkpoint files
        pp_size: Total number of pipeline stages
        tp_size: Total number of tensor parallel ranks
        using_expert_parallel: Whether expert parallelism is used
    """
    from safetensors import safe_open
    
    # Collect all weight mappings by reading the saved files
    combined_weight_map = {}
    total_size = 0
    processed_files = []
    
    for stage_idx in range(pp_size):
        # First, process the main file (from rank 0, contains non-expert weights + experts 0-31)
        main_filename = f"model-stage-{stage_idx:03d}-of-{pp_size:03d}.safetensors"
        main_filepath = os.path.join(output_dir, main_filename)
        
        if os.path.exists(main_filepath):
            # Get file size
            file_size = os.path.getsize(main_filepath)
            total_size += file_size
            processed_files.append(main_filename)
            
            # Read the safetensors file to get weight names
            try:
                with safe_open(main_filepath, framework="pt") as f:
                    # Get all tensor names in this file
                    for tensor_name in f.keys():
                        combined_weight_map[tensor_name] = main_filename
            except Exception as e:
                print(f"Warning: Could not read {main_filepath}: {e}")
                continue
        else:
            print(f"Warning: Expected file {main_filepath} not found")
        
        # If using expert parallel, also process expert files from ranks 1-7
        if using_expert_parallel and tp_size > 1:
            for tp_rank in range(1, tp_size):
                expert_filename = f"model-stage-{stage_idx:03d}-of-{pp_size:03d}-experts-tp{tp_rank:02d}.safetensors"
                expert_filepath = os.path.join(output_dir, expert_filename)
                
                if os.path.exists(expert_filepath):
                    # Get file size
                    file_size = os.path.getsize(expert_filepath)
                    total_size += file_size
                    processed_files.append(expert_filename)
                    
                    # Read the safetensors file to get weight names
                    try:
                        with safe_open(expert_filepath, framework="pt") as f:
                            # Get all tensor names in this file
                            for tensor_name in f.keys():
                                combined_weight_map[tensor_name] = expert_filename
                    except Exception as e:
                        print(f"Warning: Could not read {expert_filepath}: {e}")
                        continue
                else:
                    print(f"Warning: Expected expert file {expert_filepath} not found")
    
    # Create index structure
    index = {
        "metadata": {
            "total_size": total_size,
            "format": "pt"
        },
        "weight_map": combined_weight_map
    }
    
    # Save index file
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"Created model index at: {index_path}")
    print(f"Total model size: {total_size / (1024**3):.2f} GB")
    print(f"Total weights tracked: {len(combined_weight_map)}")
    print(f"Files processed: {len(processed_files)}")
    
    # Print breakdown of files
    main_files = [f for f in processed_files if not "experts-tp" in f]
    expert_files = [f for f in processed_files if "experts-tp" in f]
    print(f"  Main files: {len(main_files)}/{pp_size}")
    print(f"  Expert files: {len(expert_files)}/{pp_size * (tp_size - 1) if using_expert_parallel else 0}")


def merge_all_jsons_into_index(
    json_dir: str,
    temporal_dir: str,
    output_dir: str, 
    json_pattern: str = "partial_index_*.json", 
    delete_after_processing: bool = True
) -> str:
    """
    Merge all partial index JSON files into a single model.safetensors.index.json file.
    
    Args:
        json_dir: Directory containing the partial index JSON files
        temporal_dir: Path where to save the merged index (default: json_dir/model.safetensors.index.json)
        json_pattern: Pattern to match JSON files (default: "partial_index_*.json")
        output_dir: output directory to upload the merged index
        delete_after_processing: Whether to delete the JSON files after processing

    Returns:
        Path to the created index file
    """

    os.makedirs(temporal_dir, exist_ok=True)

    # Find all JSON files matching the pattern
    json_dir_patterns = os.path.join(json_dir, json_pattern)
    if json_dir.startswith('gs://'):
        # Use gsutil to download files matching the pattern to local temporal_dir
        cmd = f"gsutil -m cp -r {json_dir_patterns} {temporal_dir}"
    else:
        cmd = f"cp -r {json_dir_patterns} {temporal_dir}"
    result = subprocess.run(cmd, shell=True, text=True)

    json_files = list(glob(os.path.join(temporal_dir, json_pattern)))
    
    if not json_files:
        raise ValueError(f"No JSON files found matching pattern '{json_pattern}' in {json_dir}")
    
    print(f"Found {len(json_files)} JSON files to merge")
    
    # Set default output path if not provided
    temporal_path = Path(temporal_dir) / "model.safetensors.index.json"
    
    # Initialize aggregators
    total_size = 0
    merged_weight_map = {}
    
    # Process statistics
    processed_files = []
    skipped_files = []
    
    # Read and merge all JSON files
    for json_file in sorted(json_files):
        json_file = Path(json_file)
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract required fields
            file_size = data.get('file_size', 0)
            weight_map = data.get('weight_map', {})
            pp_rank = data.get('pp_rank', 'unknown')
            tp_rank = data.get('tp_rank', 'unknown')
            filename = data.get('filename', 'unknown')
            is_main_file = data.get('is_main_file', True)
            
            # Add to total size
            total_size += file_size
            
            # Merge weight map
            # Check for conflicts
            conflicts = set(merged_weight_map.keys()) & set(weight_map.keys())
            if conflicts:
                print(f"WARNING: Found {len(conflicts)} conflicting keys in {json_file.name}")
                print(f"  Example conflicts: {list(conflicts)[:5]}")
            
            # Update merged weight map
            merged_weight_map.update(weight_map)
            
            # Log progress
            file_size_gb = file_size / (1024**3)
            num_weights = len(weight_map)
            file_type = "main" if is_main_file else "experts"
            
            print(f"  - {json_file.name}: PP{pp_rank}/TP{tp_rank} ({file_type}) - "
                  f"{num_weights} weights, {file_size_gb:.2f} GB")
            
            processed_files.append({
                'file': json_file.name,
                'pp_rank': pp_rank,
                'tp_rank': tp_rank,
                'num_weights': num_weights,
                'size_gb': file_size_gb
            })

            os.remove(json_file)  

        except Exception as e:
            print(f"ERROR: Failed to process {json_file}: {e}")
            skipped_files.append(json_file.name)
            continue

    if delete_after_processing:
        # Delete the JSON file after processing
        if json_dir.startswith('gs://'):
            cmd = f"gsutil -m rm -r {json_dir_patterns}"
        else:
            cmd = f"rm -r {json_dir_patterns}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"All partial index JSON files deleted from {json_dir}!")

    # Create the final index structure
    final_index = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": merged_weight_map
    }
    
    # Write the merged index file
    with open(temporal_path, 'w') as f:
        json.dump(final_index, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("MERGE SUMMARY")
    print("="*50)
    print(f"Total files processed: {len(processed_files)}")
    print(f"Total files skipped: {len(skipped_files)}")
    print(f"Total weights: {len(merged_weight_map):,}")
    print(f"Total size: {total_size / (1024**4):.2f} TB ({total_size:,} bytes)")
    print(f"Output saved to: {temporal_path}")
    
    # Group by PP stage for summary
    pp_summary = {}
    for pf in processed_files:
        pp = pf['pp_rank']
        if pp not in pp_summary:
            pp_summary[pp] = {'count': 0, 'weights': 0, 'size_gb': 0}
        pp_summary[pp]['count'] += 1
        pp_summary[pp]['weights'] += pf['num_weights']
        pp_summary[pp]['size_gb'] += pf['size_gb']
    
    print("\nPer-stage summary:")
    for pp in sorted(pp_summary.keys()):
        stats = pp_summary[pp]
        print(f"  PP stage {pp}: {stats['count']} files, "
              f"{stats['weights']:,} weights, {stats['size_gb']:.2f} GB")
        
    if output_dir is not None and output_dir.startswith("gs://"):
        # Upload the merged index to output
        cmd = f"gsutil cp {str(temporal_path)} {output_dir}"
    else:
        cmd = f"cp {str(temporal_path)} {output_dir}"

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Uploaded merged index to output: {output_dir}")
        if delete_after_processing:
            os.remove(temporal_path)
            print(f"Deleted local index file: {temporal_path}")
    except Exception as e:
        print(f"Failed to upload merged index to output: {e}")

    return str(output_dir)


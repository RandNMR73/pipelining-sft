# Pipelining-SFT

**Contributors**: [Chetwin Low](https://www.linkedin.com/in/chetwin-low-061975193) [Weimin Wang](https://www.linkedin.com/in/weimin-wang-will)


A simple yet efficient full-parameter SFT framework for large-scale LLMs with MoE, supporting pipeline parallelism, expert parallelism, and tensor parallelism. This started as an experimental project—with more features coming soon—and welcomes collaboration. It is also suitable for production-scale SFT workloads.

This repository provides an end-to-end fine-tuning framework, using DeepSeek V3 as a reference implementation.

---

## 🚀 Highlights

- **Sharding**:  
  - Pipeline parallelism across nodes  
  - Expert and tensor parallelism across 8 ranks within each node

- **HuggingFace Integration**:  
  - Launch from official HuggingFace model weight directories  
  - Export back to HuggingFace checkpoint format (on GCS)

- **Mixed Precision AdamW**:  
  - Custom `mp_adamw.py` for BF16 training with FP32 master weights

- **FP8 Training (Experimental)**
  - Leveraging [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for fp8 linear layers, with custom triton kernels

- **Training Stability**:  
  - Gradient synchronization in MLA and MoE layers to prevent divergence

- **Checkpointing**:
  - BF16 checkpoint: 1.3 TB  
  - FP32 → FP8 block-quantized checkpoint: 630 GB  

- **Torch Modifications**:  
  - `torch_pipelining` contains modified files from `torch.distributed.pipelining` to support dynamic batch sizes, reduced memory bubbles, etc. 

---
## Hardware Configuration
This training framework was built with the following per-node hardware configurations. Our training cluster leverages GPUDirect-TCPX for high-speed internode communication. 

| Component  | Specification/Version       | Command to View Details  |  
|------------|-----------------------------|--------------------------| 
| GPU        | 8 x NVIDIA H100 80GB HBM3   | `nvidia-smi`             |   
| CPU        | Intel(R) Xeon(R) Platinum 8481C (96 Cores) | `lscpu` |    
| Memory     | 1.0TB DDR4                  | `free -h`                | 
| Storage    | 40TB NVMe SSD              | `df -h`                  |
| OS         | Ubuntu 22.04                | `uname -a`               | 
| CUDA       | CUDA 12.4                   | `nvcc -V`                | 
---

## Installation

### 1. Clone the repo & set up the environment:

```bash
git clone https://github.com/aaxwaz/pipelining-sft

conda create -n simple-pp python=3.11 -y
conda activate simple-pp

pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install flash_attn==2.8.1 

# install DeepGEMM
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM
git checkout 03d0be3d2d03b6eed3c99d683c0620949a13a826 # important - needs to go back to previous commit 

python setup.py install

```

---

## ⚡ Quick Start

```bash
sbatch launch.sh
```

The `launch.sh` script sets up your machines and uses SLURM to launch your job across nodes on GCP. It uses **32 nodes** by default.

Training configuration is located in `configs/train.yaml` — please adjust according to your task.

Some explanations within `configs/train.yaml`: 

- model: 
  - `model.model_dir` - download and save weights in huggingface formats https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main
  - `model.fp8_training.enabled` - if True, will enable fp8 training (experimental) 

- checkpointing: 
  - `checkpoint.use_fp8_quantization` - whether to quantize the weights to fp8 during saving. Recommended to use only for fp8 training. 
  - `checkpoint.save_dir` - directory to save the checkpoints. 
    - It can be a local path to a **NFS** (e.g. '/data/...') or a **GCS** path (e.g. 'gs://...'). 
    - **GCS** path will be recommended for stable multiprocess writing. 
  - `checkpoint.local_output_dir` - local directory used for temporally saved checkpoints and files, before being copied to `save_dir`

- distributed: 
  - `distributed.num_microbatches` - number of microbatches for PP training. Should be multiples of `pp_size`
  - `distributed.pp_size` - total number of PP stages to use. Can be 32, or 61 (i.e. each DS layer is a PP stage)
  - `total_batch_size = micro_batch_size_on_each_pp_stage × total_stages`
  - Data parallelism is currently disabled (`dp_size`==1); may be added later

---

## 📚 Lessons Learned

### 1. Mixed Precision Training

We implemented a custom **Mixed Precision AdamW** (`mp_adamw.py`) that:
- Maintains FP32 master weights while fwd, bwd and gradients are kept in bf16 precision
- Introduces minimal memory overhead
- Keeps training speed nearly unchanged
- Enables stable convergence at very low learning rates where standard AdamW fails
- DeepSeek technical report demonstrates full precision master weights is required for loss convergence, and lower precision optimizer states is sufficient, which also saves a lot of GPU memory. We empirically validated this claim.
- However, native PyTorch Adam Optimizer forces optimizer states to be in same precision as parameter dtype, which incurs high GPU consumption, thus to support lower precision optimizer states with full precision master weights, we made small but vital changes to PyTorch's official AdamW implementation (Torch 2.6) 

#### 📈 Training Stability Comparison:

![MPAdamW vs AdamW](graphs/compare_mp_adamw.png)

> Green = MPAdamW, Orange = BF16 AdamW  
> Under a small learning rate (`1e-7`), standard AdamW with BF16 fails to converge, while MPAdamW converges smoothly with minimal training slowdown and memory increase.

### 2. Gradient Synchronization for Stability

Added `grad_sync` functions in both **MLA** and **MoE** layers, helping to prevent training divergence across distributed ranks.

In our development of our multi-parallelism training framework, we noticed some interesting properties of Deepseek's MLA and MoEs trained with Expert Parallel, specifically related to gradient divergence in the backward if not handled precisely.

1. MLA: It uses shared learnable position embeddings (k_pe) across attention heads for K projections, which means that when Tensor-Parallel (TP) is enabled for attention, k_pe receives different gradients in the backward pass for different sets of attention heads and thus if we do not add backward hooks to sync gradients for k_pe, the gradients will start to diverge across replicated parameters within the TP group.

2. MoE: Most MoE forward implementations use indices masking to efficiently run experts on selected tokens and accumulate the results coupled with the corresponding expert weight in an output tensor. Due to the non-determinism of PyTorch's matrix operations, there will be differences in gradients for replicated parameters within Expert Parallel group. Hence we require 2 explicit gradient hooks in the backward for a single MoE layer.

An alternative implementation would be to explicitly sync all replicated parameters within the TP and/or EP group at the end of the backward, but we chose to implement with grad_sync since it looks cleaner.

![w/ and w/o grad_sync](graphs/compare_grad_sync.png)

> Green = with grad_sync, Orange = not using grad_sync  
> without using grad_sync, training is becoming unstable with grad_norm exploding and loss curves a bit higher

### 3. Training Speed (H100 with GPUDirect-TCPX* Benchmark) on 32 nodes

| Total Batch Size | Context Length | Iteration Time |
|------------------|----------------|----------------|
| 128              | 2k             | 27s            |
| 64               | 4k             | 27s            |

*Slower than InfiniBand

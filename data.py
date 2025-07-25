import torch 
from torch.utils.data import DataLoader
from datasets import load_dataset
from train_utils import Config 
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm 
import torch.distributed as dist

class UltraChatDataset(torch.utils.data.Dataset):
    """UltraChat dataset for supervised fine-tuning"""
    
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        messages = example['messages']
        
        # Format messages for chat template
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (shift by 1 for causal LM)
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

class MMLUDataset(torch.utils.data.Dataset):
    """MMLU dataset for evaluation"""
    
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Answer mapping
        self.answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.answer_to_id = {
            "A": tokenizer.encode("A", add_special_tokens=False)[0],
            "B": tokenizer.encode("B", add_special_tokens=False)[0],
            "C": tokenizer.encode("C", add_special_tokens=False)[0],
            "D": tokenizer.encode("D", add_special_tokens=False)[0],
        }
        
    def __len__(self):
        return len(self.dataset)
    
    def format_question(self, example):
        """Format MMLU question in multiple choice format"""
        question = example['question']
        choices = example['choices']
        
        # Format as multiple choice
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{self.answer_map[i]}) {choice}\n"
        prompt += "Answer:"
        
        return prompt
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Format the question
        prompt = self.format_question(example)
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Get the correct answer token ID
        correct_answer = self.answer_map[example['answer']]
        answer_token_id = self.answer_to_id[correct_answer]
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'answer_token_id': answer_token_id,
            'subject': example.get('subject', 'unknown'),
        }


def create_dataloader(config: Config, tokenizer, is_train: bool = True):
    """Create dataloader for pipeline parallel training - all ranks see same data"""

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Only rank 0 downloads and processes the dataset
    if rank == 0:
        dataset = load_dataset(
            config.dataset['name'],
            split=config.dataset['train_split'] if is_train else config.dataset['eval_split'],
            cache_dir=config.dataset.get('cache_dir', "/mnt/localdisk/dataloader_cache")
        )
        print(f"Rank 0: Successfully loaded {'train' if is_train else 'eval'} dataset")

    if dist.is_initialized():
        dist.barrier()
    
    # Now all other ranks can safely load from cache
    if rank != 0:
        dataset = load_dataset(
            config.dataset['name'],
            split=config.dataset['train_split'] if is_train else config.dataset['eval_split'],
            cache_dir=config.dataset.get('cache_dir', "/mnt/localdisk/dataloader_cache")
        )
    
    # Limit samples if specified
    if is_train and config.dataset.get('max_train_samples'):
        dataset = dataset.select(range(min(len(dataset), config.dataset['max_train_samples'])))
    elif not is_train and config.dataset.get('max_eval_samples'):
        dataset = dataset.select(range(min(len(dataset), config.dataset['max_eval_samples'])))
    
    # Create dataset wrapper
    dataset = UltraChatDataset(dataset, tokenizer, config.training['max_sequence_length'])
    
    # Create regular dataloader WITHOUT DistributedSampler
    # All ranks will see the same batches in the same order
    dataloader = DataLoader(
        dataset,
        batch_size=config.training['total_batch_size'] if is_train else config.training['eval_batch_size'],
        shuffle=is_train,  # Same random seed across ranks ensures same shuffle
        num_workers=config.dataset['num_workers'],
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    return dataloader

def create_mmlu_dataloader(config: Config, tokenizer):
    """Create MMLU evaluation dataloader"""

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        # Load MMLU dataset
        dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=config.dataset.get('cache_dir', "/mnt/localdisk/dataloader_cache"))

    if dist.is_initialized():
        dist.barrier()
    
    if rank != 0:
        # Other ranks load from cache
        dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=config.dataset.get('cache_dir', "/mnt/localdisk/dataloader_cache"))
    
    # Optionally limit the number of samples for faster evaluation
    max_samples = config.training.get('mmlu_max_samples', 1000)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    # Create dataset wrapper
    mmlu_dataset = MMLUDataset(dataset, tokenizer, config.training['max_sequence_length'])
    
    # Create dataloader
    dataloader = DataLoader(
        mmlu_dataset,
        batch_size=config.training['eval_batch_size'],
        shuffle=False,
        num_workers=config.dataset['num_workers'],
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    return dataloader


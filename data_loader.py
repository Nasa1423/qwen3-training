"""
Data loader for JSON dialog datasets.
Converts dialogs to Qwen3 chat format.
"""
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DialogDataset(Dataset):
    """Dataset for dialog fine-tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        data = self._load_json(data_path)
        
        # Shuffle and split
        import random
        random.seed(seed)
        random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        if split == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]
        
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON file with dialogs."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate format
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of dialog objects")
        
        valid_data = []
        for item in data:
            if "question" in item and "answer" in item:
                valid_data.append(item)
        
        print(f"Loaded {len(valid_data)} valid dialogs from {path}")
        return valid_data
    
    def _format_dialog(self, item: Dict[str, Any]) -> str:
        """Format dialog entry to Qwen3 chat format."""
        messages = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]}
        ]
        
        # Use tokenizer's chat template (without thinking for Qwen3)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False  # Disable thinking mode
        )
        return text
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = self._format_dialog(item)
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Labels = input_ids (causal LM), mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_datasets(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple:
    """Create train and eval datasets."""
    train_dataset = DialogDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
    )
    
    eval_dataset = DialogDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="eval",
        train_ratio=train_ratio,
        seed=seed,
    )
    
    return train_dataset, eval_dataset


def get_data_collator(tokenizer: AutoTokenizer):
    """Simple data collator for causal LM."""
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

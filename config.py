"""
Training configuration for Qwen3-4B fine-tuning.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen3-4B"
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    trust_remote_code: bool = True
    

@dataclass
class LoraConfig:
    """LoRA configuration."""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "./data/dialogs.json"
    max_length: int = 2048
    train_split: float = 0.9
    shuffle: bool = True
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Reporting
    report_to: str = "tensorboard"  # tensorboard, wandb, none
    run_name: Optional[str] = None
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class FullConfig:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

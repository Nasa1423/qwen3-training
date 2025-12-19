"""
Full fine-tuning script for Qwen3-4B.
Supports multi-GPU training via Accelerate.

Usage:
    # Single GPU
    python train_full.py --data_path ./data/dialogs.json
    
    # Multi-GPU with accelerate
    accelerate launch train_full.py --data_path ./data/dialogs.json
"""
import argparse
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from accelerate import Accelerator

from data_loader import create_datasets, get_data_collator
from config import ModelConfig, DataConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Full fine-tuning for Qwen3-4B")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON dialogs file")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train/eval split ratio")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name or path")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention 2")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="./output/full", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 mixed precision")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Qwen3-4B Full Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("\n[2/4] Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
    }
    
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Create datasets
    print("\n[3/4] Loading datasets...")
    train_dataset, eval_dataset = create_datasets(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_ratio=args.train_split,
        seed=args.seed,
    )
    
    data_collator = get_data_collator(tokenizer)
    
    # Training arguments
    print("\n[4/4] Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to if args.report_to != "none" else None,
        run_name=args.run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {output_dir / 'final'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

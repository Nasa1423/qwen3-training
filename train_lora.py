"""
LoRA fine-tuning script for Qwen3-4B.
Memory-efficient training using PEFT.

Usage:
    # Single GPU
    python train_lora.py --data_path ./data/dialogs.json
    
    # Multi-GPU with accelerate
    accelerate launch train_lora.py --data_path ./data/dialogs.json
    
    # Merge LoRA weights after training
    python train_lora.py --merge_adapter --adapter_path ./output/lora/final
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
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

from data_loader import create_datasets, get_data_collator


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-4B")
    
    # Data
    parser.add_argument("--data_path", type=str, default=None, help="Path to JSON dialogs file")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train/eval split ratio")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name or path")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention 2")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true", default=False, help="Use 8-bit quantization")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="./output/lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
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
    
    # Merge mode
    parser.add_argument("--merge_adapter", action="store_true", help="Merge LoRA adapter into base model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter for merging")
    parser.add_argument("--merged_output", type=str, default="./output/merged", help="Output path for merged model")
    
    return parser.parse_args()


def get_quantization_config(args):
    """Get BitsAndBytesConfig for quantization."""
    if args.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def merge_adapter(args):
    """Merge LoRA adapter into base model."""
    print("=" * 60)
    print("Merging LoRA adapter into base model")
    print("=" * 60)
    
    if not args.adapter_path:
        raise ValueError("--adapter_path is required for merging")
    
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    print(f"Adapter: {adapter_path}")
    print(f"Base model: {args.model_name}")
    
    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Load adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    
    # Merge
    print("Merging weights...")
    model = model.merge_and_unload()
    
    # Save
    output_path = Path(args.merged_output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    print("\n" + "=" * 60)
    print("Merge completed!")
    print("=" * 60)


def train(args):
    """Main training function."""
    set_seed(args.seed)
    
    if not args.data_path:
        raise ValueError("--data_path is required for training")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Qwen3-4B LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    if args.use_4bit:
        print("Quantization: 4-bit (QLoRA)")
    elif args.use_8bit:
        print("Quantization: 8-bit")
    else:
        print("Quantization: None (BF16)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("\n[2/5] Loading model...")
    quant_config = get_quantization_config(args)
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }
    
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["device_map"] = "auto"
    
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # Prepare for k-bit training if using quantization
    if quant_config:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("\n[3/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled")
    
    # Create datasets
    print("\n[4/5] Loading datasets...")
    train_dataset, eval_dataset = create_datasets(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_ratio=args.train_split,
        seed=args.seed,
    )
    
    data_collator = get_data_collator(tokenizer)
    
    # Training arguments
    print("\n[5/5] Setting up training...")
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
    print("Starting LoRA training...")
    print("=" * 60)
    
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    
    # Save final adapter
    print("\nSaving LoRA adapter...")
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"LoRA adapter saved to: {final_path}")
    print(f"\nTo merge adapter into base model, run:")
    print(f"  python train_lora.py --merge_adapter --adapter_path {final_path}")
    print("=" * 60)


def main():
    args = parse_args()
    
    if args.merge_adapter:
        merge_adapter(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

"""
Inference script for fine-tuned Qwen3-4B models.
Supports both full fine-tuned and LoRA adapter models.

Usage:
    # Full fine-tuned model
    python inference.py --model_path ./output/full/final
    
    # LoRA adapter
    python inference.py --model_path Qwen/Qwen3-4B --adapter_path ./output/lora/final
    
    # Batch inference from file
    python inference.py --model_path ./output/full/final --input_file questions.txt --output_file answers.txt
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Qwen3-4B")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or base model name")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--use_flash_attn", action="store_true", default=True, help="Use Flash Attention 2")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling")
    
    # Mode
    parser.add_argument("--interactive", action="store_true", default=False, help="Interactive chat mode")
    parser.add_argument("--input_file", type=str, default=None, help="File with questions (one per line)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for answers")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for generation")
    
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer."""
    print(f"Loading model: {args.model_path}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter if specified
    if args.adapter_path:
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, args) -> str:
    """Generate response for a given prompt."""
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # No thinking mode
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def interactive_mode(model, tokenizer, args):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            response = generate_response(model, tokenizer, user_input, args)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(model, tokenizer, args):
    """Batch inference from file."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(questions)} questions...")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing...")
        response = generate_response(model, tokenizer, question, args)
        results.append({
            "question": question,
            "answer": response
        })
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")
    else:
        for r in results:
            print(f"\nQ: {r['question']}")
            print(f"A: {r['answer']}")
    
    return results


def single_prompt(model, tokenizer, args):
    """Generate response for a single prompt."""
    response = generate_response(model, tokenizer, args.prompt, args)
    print(f"\nPrompt: {args.prompt}")
    print(f"\nResponse: {response}")
    return response


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args)
    
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.input_file:
        batch_inference(model, tokenizer, args)
    elif args.prompt:
        single_prompt(model, tokenizer, args)
    else:
        # Default to interactive mode
        interactive_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()

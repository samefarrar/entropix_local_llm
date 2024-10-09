import argparse
import time
import mlx.core as mx
import pathlib
from typing import Optional, List, Tuple, Union
from mlx_model import load_entropix_model
from mlx_lm.utils import load
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import generate as generate_mlx_lm
from mlx_generate import generate
from mlx_sampler import SamplerConfig
from mlx_prompts import create_prompts_from_csv, prompt1, prompt2, prompt3, prompt4, prompt5, thinking_prompt, o1_claude_prompt
import inspect
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description = "Generate text using Entropy based sampling based on input prompts")
    parser.add_argument("--prompts", action="store_true", help = "Use predefined prompts from mlx_entropix.prompts")
    parser.add_argument("--prompt_csv", action="store_true", help = "Use prompts from data/prompts.csv")
    parser.add_argument("--input", type = str, help = "Input prompt to generate text")
    parser.add_argument("--entropix", action="store_true", default=True, help="Use Entropix model for generation")
    parser.add_argument("--normal", action="store_true", help="Use normal model for generation")
    args = parser.parse_args()


    if not args.prompts and not args.input and not args.prompt_csv:
        print("No input provided. Use --prompts to use predefined prompts from mlx_entropix.prompts or provide a custom prompt using --input")
        print("Exiting...")
        exit()

    if args.normal:
        model, tokenizer = load("weights/Llama-3.2-1B-Instruct")
        model_with_scores = False
        sample_config_kwargs = SamplerConfig()
        sample_config_kwargs = {
            "temp": sample_config_kwargs.temp,
            "top_p": sample_config_kwargs.top_p,
            "min_p": sample_config_kwargs.min_p,
        }
    else:
        path = Path("weights/Llama-3.2-1B-Instruct")
        tokenizer = load_tokenizer(path)
        model = load_entropix_model(path)
        model_with_scores = True
    max_tokens = 4096

    if args.input:
        messages = [
            {"role": "system", "content": thinking_prompt},
            {"role": "user", "content": args.input}
        ]
        prompt = thinking_prompt.format(prompt = args.input)
        if args.normal:
            response = generate_mlx_lm(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens, **sample_config_kwargs)
        else:
            response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens)
    elif args.prompt_csv:
        prompts = create_prompts_from_csv("data/acp.csv")
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if args.normal:
                response = generate_mlx_lm(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens, **sample_config_kwargs)
            else:
                response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens)
    elif args.prompts:
        prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if args.normal:
                response = generate_mlx_lm(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens, **sample_config_kwargs)
            else:
                response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens = max_tokens)

if __name__ == "__main__":
    main()

import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    
    # 只加载 Tokenizer，不加载 PyTorch 模型（太慢了！）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 返回 None 作为 model，因为我们不需要它
    return tokenizer, None, model_path


def load_llaisys_model(model_path, device_name):
    # 这里的 load_weights 我们在 Python 包装里实现了
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    model.load_weights() # 显式调用加载权重
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    print("\n[Test] Tokenizing prompt...")
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    print(f"[Test] Input Tokens: {inputs}")
    
    print(f"[Test] Starting LLAISYS generation (max {max_new_tokens} tokens)...")
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    # 1. 这一步很快，只加载分词器
    tokenizer, _, model_path = load_hf_model(args.model, args.device)

    # === SKIPPING PYTORCH INFERENCE ===
    # 我们把原来这里的 hf_infer 全删了，直接进正题
    
    print("\n=== LLAISYS Inference ===\n")
    
    # 2. 加载你的 C++ 模型
    model = load_llaisys_model(model_path, args.device)
    
    start_time = time.time()
    
    # 3. 运行你的推理
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps, # 实际上受限于 qwen2.py 里的硬编码 5
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    end_time = time.time()

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")
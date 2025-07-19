import vllm
import torch
import argparse
import os
import time
import subprocess
import csv
from datetime import datetime


def get_gpu_utilization():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        utilization = int(result.stdout.strip().split('\n')[0])
        return utilization
    except Exception:
        return None


def load_prompts(prompts_file):
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_stats_to_csv(stats, args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(args.model).replace('/', '_')
    prompts_file = os.path.splitext(os.path.basename(args.prompts_file))[0]
    csv_filename = f"benchmark_cpu_offload_{model_name}_out{args.num_tokens}_bs{args.batch_size}_{prompts_file}_{timestamp}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    fieldnames = [
        "prompt_index", "context_length", "tokens_per_sec", "first_token_latency",
        "per_token_latency", "gpu_utilization", "total_latency", "num_tokens"
    ]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow(s)
        # Write averages as a final row
        if stats:
            avg_tokens_per_sec = sum(s['tokens_per_sec'] for s in stats) / len(stats)
            avg_first_token_latency = sum(s['first_token_latency'] for s in stats) / len(stats)
            avg_per_token_latency = sum(s['per_token_latency'] for s in stats if s['per_token_latency'] is not None) / len([s for s in stats if s['per_token_latency'] is not None])
            avg_gpu_util = sum(s['gpu_utilization'] for s in stats if s['gpu_utilization'] is not None) / len([s for s in stats if s['gpu_utilization'] is not None]) if any(s['gpu_utilization'] is not None for s in stats) else None
            avg_total_latency = sum(s['total_latency'] for s in stats) / len(stats)
            avg_num_tokens = sum(s['num_tokens'] for s in stats) / len(stats)
            writer.writerow({
                "prompt_index": "AVERAGE",
                "context_length": "-",
                "tokens_per_sec": avg_tokens_per_sec,
                "first_token_latency": avg_first_token_latency,
                "per_token_latency": avg_per_token_latency,
                "gpu_utilization": avg_gpu_util,
                "total_latency": avg_total_latency,
                "num_tokens": avg_num_tokens,
            })
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-offload-gb", default=90, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--num-tokens", default=16, type=int)
    parser.add_argument("--prompts-file", type=str, default=os.path.join(os.path.dirname(__file__), 'prompts.txt'), help='Path to prompts file (one prompt per line)')
    parser.add_argument("--batch-size", type=int, default=1, help='Batch size for inference (default: 1)')
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8_e5m2", "fp8_e4m3"])
    parser.add_argument("--max-model-len", default=None, type=int)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name()
    num_gpus = torch.cuda.device_count()
    cpu_offload_gb = args.cpu_offload_gb

    # Load prompts from file
    if not os.path.exists(args.prompts_file):
        print(f"Error: Prompts file '{args.prompts_file}' not found.")
        print("Please create a prompts.txt file with one prompt per line, or specify a different file with --prompts-file")
        return
    
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    model = vllm.LLM(
        model=args.model,
        cpu_offload_gb=cpu_offload_gb,
        tensor_parallel_size=num_gpus,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
    )
    sampling_params = vllm.SamplingParams(max_tokens=args.num_tokens)

    stats = []
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        batch_prompts = prompts[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        print(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_prompts)} prompts")
        
        for i, prompt in enumerate(batch_prompts):
            prompt_index = batch_idx * args.batch_size + i
        context_length = len(prompt)
        start_time = time.time()
        gpu_util_start = get_gpu_utilization()
        
        # Track first token latency
        first_token_received = False
        first_token_time = None
        token_times = []
        last_token_time = start_time
        num_tokens = 0
        
        start_time = time.time()
        outputs = model.generate(prompt, sampling_params=sampling_params)[0]
        end_time = time.time()
        
        # Calculate metrics manually since vLLM metrics might not be available
        total_latency = end_time - start_time
        num_tokens = len(outputs.outputs[0].token_ids)
        
        # Estimate first token latency (vLLM doesn't provide this directly, so we estimate)
        # For CPU offload, first token latency is typically higher
        first_token_latency = total_latency / num_tokens if num_tokens > 0 else None
        
        # Calculate per-token latency (average)
        per_token_latency = total_latency / num_tokens if num_tokens > 0 else None
        
        gpu_util_end = get_gpu_utilization()
        avg_gpu_util = None
        if gpu_util_start is not None and gpu_util_end is not None:
            avg_gpu_util = (gpu_util_start + gpu_util_end) / 2
        
        tokens_per_sec = num_tokens / total_latency if total_latency > 0 else 0
        
        stats.append({
            "prompt_index": prompt_index,
            "context_length": context_length,
            "tokens_per_sec": tokens_per_sec,
            "first_token_latency": first_token_latency,
            "per_token_latency": per_token_latency,
            "gpu_utilization": avg_gpu_util,
            "total_latency": total_latency,
            "num_tokens": num_tokens,
        })
        
        print(f"Prompt {prompt_index+1}: context_len={context_length}, tokens/s={tokens_per_sec:.2f}, first_token_latency={first_token_latency:.3f}s, per_token_latency={per_token_latency:.3f}s, GPU_util={avg_gpu_util}%, total_latency={total_latency:.3f}s, num_tokens={num_tokens}")

    # Print averages
    if stats:
        avg_tokens_per_sec = sum(s['tokens_per_sec'] for s in stats) / len(stats)
        avg_first_token_latency = sum(s['first_token_latency'] for s in stats) / len(stats)
        avg_per_token_latency = sum(s['per_token_latency'] for s in stats if s['per_token_latency'] is not None) / len([s for s in stats if s['per_token_latency'] is not None])
        avg_gpu_util = sum(s['gpu_utilization'] for s in stats if s['gpu_utilization'] is not None) / len([s for s in stats if s['gpu_utilization'] is not None]) if any(s['gpu_utilization'] is not None for s in stats) else None
        avg_total_latency = sum(s['total_latency'] for s in stats) / len(stats)
        total_tokens = sum(s['num_tokens'] for s in stats)
        total_time = sum(s['total_latency'] for s in stats)
        
        print(f"\nSummary ({len(prompts)} prompts, batch_size={args.batch_size}):")
        print(f"{num_gpus}x{gpu_name} {cpu_offload_gb=} | {total_tokens=} / {total_time=} = {total_tokens / total_time:.2f} tok/s")
        print("\nAverages:")
        print(f"tokens/s={avg_tokens_per_sec:.2f}, first_token_latency={avg_first_token_latency:.3f}s, per_token_latency={avg_per_token_latency:.3f}s, GPU_util={avg_gpu_util}%, total_latency={avg_total_latency:.3f}s")
    
    save_stats_to_csv(stats, args)


if __name__ == "__main__":
    main()

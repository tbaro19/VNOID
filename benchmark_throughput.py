import time
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import argparse
import subprocess
import re
import os
from math import ceil
import csv
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM throughput.")
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--output-len', type=int, default=100, help='Number of output tokens (default: 100)')
    parser.add_argument('--prompts-file', type=str, default=os.path.join(os.path.dirname(__file__), 'prompts.txt'), help='Path to prompts file (one prompt per line)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference (default: 1)')
    return parser.parse_args()

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

async def benchmark_model(model_name, output_len, prompts, batch_size):
    engine_args = AsyncEngineArgs(model=model_name)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    stats = []
    sampling_params = SamplingParams(max_tokens=output_len)
    num_batches = ceil(len(prompts) / batch_size)
    
    for batch_idx in range(num_batches):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        context_lengths = [len(p) for p in batch_prompts]
        start_time = time.time()
        gpu_util_start = get_gpu_utilization()
        batch_first_token_times = [None] * len(batch_prompts)
        batch_first_token_received = [False] * len(batch_prompts)
        batch_token_times = [[] for _ in batch_prompts]
        batch_num_tokens = [0] * len(batch_prompts)
        batch_last_token_time = [start_time] * len(batch_prompts)
        request_ids = [f"req_{batch_idx}_{i}" for i in range(len(batch_prompts))]
        reqid_to_idx = {rid: i for i, rid in enumerate(request_ids)}
        gens = [engine.generate(p, sampling_params, request_id=rid) for p, rid in zip(batch_prompts, request_ids)]
        tasks = [consume_output(gen, idx, batch_first_token_times, batch_first_token_received, batch_token_times, batch_last_token_time, batch_num_tokens, start_time, reqid_to_idx, rid) for idx, (gen, rid) in enumerate(zip(gens, request_ids))]
        await asyncio.gather(*tasks)
        total_latency = time.time() - start_time
        gpu_util_end = get_gpu_utilization()
        avg_gpu_util = None
        if gpu_util_start is not None and gpu_util_end is not None:
            avg_gpu_util = (gpu_util_start + gpu_util_end) / 2
        for i, prompt in enumerate(batch_prompts):
            tokens_per_sec = batch_num_tokens[i] / total_latency if total_latency > 0 else 0
            per_token_latency = sum(batch_token_times[i]) / len(batch_token_times[i]) if batch_token_times[i] else None
            stats.append({
                "prompt_index": batch_idx * batch_size + i,
                "context_length": context_lengths[i],
                "tokens_per_sec": tokens_per_sec,
                "first_token_latency": batch_first_token_times[i],
                "per_token_latency": per_token_latency,
                "gpu_utilization": avg_gpu_util,
                "total_latency": total_latency,
                "num_tokens": batch_num_tokens[i],
            })
    return stats

async def consume_output(gen, idx, batch_first_token_times, batch_first_token_received, batch_token_times, batch_last_token_time, batch_num_tokens, start_time, reqid_to_idx, rid):
    async for output in gen:
        now = time.time()
        if not batch_first_token_received[idx]:
            batch_first_token_times[idx] = now - start_time
            batch_first_token_received[idx] = True
        else:
            batch_token_times[idx].append(now - batch_last_token_time[idx])
        batch_last_token_time[idx] = now
        batch_num_tokens[idx] += 1

def save_stats_to_csv(stats, args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(args.model).replace('/', '_')
    prompts_file = os.path.splitext(os.path.basename(args.prompts_file))[0]
    csv_filename = f"benchmark_{model_name}_out{args.output_len}_bs{args.batch_size}_{prompts_file}_{timestamp}.csv"
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

if __name__ == "__main__":
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    stats = asyncio.run(benchmark_model(args.model, args.output_len, prompts, args.batch_size))
    print("\nBenchmark Results:")
    for s in stats:
        print(f"Prompt {s['prompt_index']}: context_len={s['context_length']}, tokens/s={s['tokens_per_sec']:.2f}, first_token_latency={s['first_token_latency']:.3f}s, per_token_latency={s['per_token_latency']:.3f}s, GPU_util={s['gpu_utilization']}%, total_latency={s['total_latency']:.3f}s, num_tokens={s['num_tokens']}")
    # Optionally, print averages
    if stats:
        avg_tokens_per_sec = sum(s['tokens_per_sec'] for s in stats) / len(stats)
        avg_first_token_latency = sum(s['first_token_latency'] for s in stats) / len(stats)
        avg_per_token_latency = sum(s['per_token_latency'] for s in stats if s['per_token_latency'] is not None) / len([s for s in stats if s['per_token_latency'] is not None])
        avg_gpu_util = sum(s['gpu_utilization'] for s in stats if s['gpu_utilization'] is not None) / len([s for s in stats if s['gpu_utilization'] is not None]) if any(s['gpu_utilization'] is not None for s in stats) else None
        avg_total_latency = sum(s['total_latency'] for s in stats) / len(stats)
        print("\nAverages:")
        print(f"tokens/s={avg_tokens_per_sec:.2f}, first_token_latency={avg_first_token_latency:.3f}s, per_token_latency={avg_per_token_latency:.3f}s, GPU_util={avg_gpu_util}%, total_latency={avg_total_latency:.3f}s")
    save_stats_to_csv(stats, args)
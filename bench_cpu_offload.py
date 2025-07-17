import vllm
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-offload-gb", default=90, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--num-tokens", default=16, type=int)
    parser.add_argument("--prompt", default="Here is a 5000 word essay on the best color of leaves during the fall:")
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8_e5m2", "fp8_e4m3"])
    parser.add_argument("--max-model-len", default=None, type=int)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name()
    num_gpus = torch.cuda.device_count()
    cpu_offload_gb = args.cpu_offload_gb

    model = vllm.LLM(
        model=args.model,
        cpu_offload_gb=cpu_offload_gb,
        tensor_parallel_size=num_gpus,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
    )
    sampling_params = vllm.SamplingParams(max_tokens=args.num_tokens)

    outputs = model.generate(args.prompt, sampling_params=sampling_params)[0]

    num_tokens = len(outputs.outputs[0].token_ids)
    total_s = outputs.metrics.finished_time - outputs.metrics.arrival_time
    print(f"{num_gpus}x{gpu_name} {cpu_offload_gb=} | {num_tokens=} / {total_s=} = {num_tokens / total_s:.2f} tok/s")

if __name__ == "__main__":
    main()

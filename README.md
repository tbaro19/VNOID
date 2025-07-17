# VNOID
Scripts for Vietnam OpenInfra &amp;  Cloud Native Day 2025 - Topic: ARM64 Meets AI: Benchmarking LLM Performance and Efficiency Across GH200, H100 and A100

This directory contains scripts to benchmark the performance of large language models (LLMs) using [vLLM](https://github.com/vllm-project/vllm). These scripts help you measure throughput, latency, and the effect of CPU offloading on inference.

## Requirements

- Python 3.8+
- [vLLM](https://github.com/vllm-project/vllm)
- [PyTorch](https://pytorch.org/)
- NVIDIA GPU (for GPU benchmarks)
- CUDA drivers (for GPU benchmarks)

Install dependencies:
```bash
pip install vllm torch
```

---

## 1. `benchmark_throughput.py`

**Purpose:**
- Benchmark LLM inference throughput, first token latency, per-token latency, GPU utilization, and total request latency.
- Supports batching and custom prompt files.

**Usage:**
```bash
python3 benchmark_throughput.py \
  --model <model_name_or_path> \
  --output-len <num_output_tokens> \
  --prompts-file <path_to_prompts.txt> \
  --batch-size <batch_size>
```

**Arguments:**
- `--model`: Model name or path (e.g., `moonshotai/Kimi-K2-Instruct`, `Menlo/Jan-nano`)
- `--output-len`: Number of output tokens to generate per prompt (default: 100)
- `--prompts-file`: Path to a text file with one prompt per line (default: `prompts.txt` in this directory)
- `--batch-size`: Number of prompts to process in parallel (default: 1)

**Example:**
```bash
python3 benchmark_throughput.py --model Menlo/Jan-nano --output-len 100 --prompts-file prompts.txt --batch-size 1
```

**Output:**
- Prints per-prompt and average statistics (tokens/sec, latency, GPU utilization, etc.)
- Saves results to a CSV file in the current directory.

---

## 2. `bench_cpu_offload.py`

**Purpose:**
- Benchmark LLM inference with CPU offloading, useful for running large models on limited GPU memory.
- Measures throughput (tokens/sec) with configurable CPU offload size.

**Usage:**
```bash
python3 bench_cpu_offload.py \
  --model <model_name_or_path> \
  --cpu-offload-gb <GB_to_offload_to_CPU> \
  --num-tokens <output_tokens> \
  --prompt <prompt_text> \
  --kv-cache-dtype <auto|fp8_e5m2|fp8_e4m3> \
  --max-model-len <max_context_length>
```

**Arguments:**
- `--model`: Model name or path (default: `meta-llama/Llama-3.1-70B`)
- `--cpu-offload-gb`: Amount of model weights (in GB) to offload to CPU (default: 90)
- `--num-tokens`: Number of output tokens to generate (default: 16)
- `--prompt`: Prompt text for generation (default: a long essay prompt)
- `--kv-cache-dtype`: Data type for KV cache (`auto`, `fp8_e5m2`, or `fp8_e4m3`; default: `auto`)
- `--max-model-len`: Maximum context length (optional)

**Example:**
```bash
python3 bench_cpu_offload.py --model Menlo/Jan-nano --cpu-offload-gb 40 --num-tokens 32 --prompt "Write a poem about AI."
```

**Output:**
- Prints the number of tokens generated, total time taken, and throughput (tokens/sec).

---

## Prompts File Format
- For `benchmark_throughput.py`, provide a text file (e.g., `prompts.txt`) with one prompt per line.

---

## References
- [vLLM Documentation](https://vllm.readthedocs.io/en/latest/)
- [Menlo/Jan-nano Model Card](https://huggingface.co/Menlo/Jan-nano)
- [MoonshotAI/Kimi-K2-Instruct Model Card](https://huggingface.co/moonshotai/Kimi-K2-Instruct)

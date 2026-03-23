import json
import time
from dataclasses import asdict, dataclass
from enum import Enum

import llama_cpp
import numpy as np
from faker import Faker
from zeus.monitor.energy import Measurement, ZeusMonitor

# ============================================================================
# Type Definitions
# ============================================================================


@dataclass
class SystemMeasurement:
    cpu: float | None = None
    gpu: float | None = None
    dram: float | None = None


@dataclass
class EnergyMeasurement:
    pp_tokens: int
    gen_tokens: int
    pp_time: float
    gen_time: float
    pp_j: SystemMeasurement
    gen_j: SystemMeasurement
    idle_power_w: SystemMeasurement


class Quantization(Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"

    def to_mlx_model_name(self) -> str:
        if self == Quantization.FP16:
            return "mlx-community/Qwen2.5-7B-Instruct-FP16"
        elif self == Quantization.INT8:
            return "mlx-community/Qwen2.5-7B-Instruct-8bit"
        elif self == Quantization.INT4:
            return "mlx-community/Qwen2.5-7B-Instruct-4bit"
        else:
            raise ValueError(f"Unsupported quantization: {self}")

    def to_ttuf_model_name(self) -> str:
        if self == Quantization.FP16:
            return "models/qwen2.5-7b/Qwen2.5-7B-Instruct-f16.gguf"
        elif self == Quantization.INT8:
            return "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q8_0.gguf"
        elif self == Quantization.INT4:
            return "models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        else:
            raise ValueError(f"Unsupported quantization: {self}")


# ============================================================================
# Helper Functions
# ============================================================================


def metrics_to_joules(m: Measurement) -> SystemMeasurement:
    """Convert a ZeusMonitor measurement to total joules."""

    return SystemMeasurement(
        cpu=sum(m.cpu_energy.values()) if m.cpu_energy else None,
        gpu=sum(m.gpu_energy.values()) if m.gpu_energy else None,
        dram=sum(m.dram_energy.values()) if m.dram_energy else None,
    )


def measure_idle_power(monitor: ZeusMonitor) -> SystemMeasurement:
    """Measure mean idle CPU+GPU power (W) over IDLE_SECONDS."""
    monitor.begin_window("idle")
    time.sleep(IDLE_SECONDS)
    m = monitor.end_window("idle")
    idle_j = metrics_to_joules(m)
    return SystemMeasurement(
        cpu=(idle_j.cpu or 0) / IDLE_SECONDS,
        gpu=(idle_j.gpu or 0) / IDLE_SECONDS,
        dram=(idle_j.dram or 0) / IDLE_SECONDS,
    )


def generate_random_token_prompt(
    tokenizer, num_tokens: int, seed: int | None = None
) -> list[int]:
    """Generate a random prompt of the specified token length using the tokenizer's vocab."""
    fake = Faker()
    if seed is not None:
        fake.seed_instance(seed)
    text_buffer = ""
    while len(text_buffer.split()) < 2 * num_tokens:
        text_buffer += fake.sentence() + " "
    return tokenizer.encode(text_buffer)[:num_tokens]


# ============================================================================
# Core Inference & Benchmarking
# ============================================================================


def _llama_cuda_single_inference(
    llm: llama_cpp.Llama,
    prompt: str | list[int],
    prompt_tokens: int,
    monitor: ZeusMonitor,
) -> EnergyMeasurement:
    """Single llama-cpp inference with sequential prefill/generation energy windows.

    Returns:
        (prompt_tokens, completion_tokens, request_start,
         first_token_time, generation_end_time, prefill_raw_j, gen_raw_j)
    """
    idle_power_w = measure_idle_power(monitor)
    prefill_metrics = None
    gen_metrics = None
    first_token_time = None
    generation_end_time = None
    output_chunks = []

    monitor.begin_window("prefill")
    request_start = time.perf_counter()
    for chunk in llm(prompt, max_tokens=GENERATION_TOKENS, temperature=0, stream=True):  # ty:ignore[invalid-argument-type]
        text = chunk["choices"][0]["text"]  # ty:ignore[invalid-argument-type]
        if not text:
            continue
        if first_token_time is None:
            first_token_time = time.perf_counter()
            prefill_metrics = monitor.end_window("prefill")
            monitor.begin_window("generation")
        output_chunks.append(text)
        generation_end_time = time.perf_counter()

    if first_token_time is None:
        # No tokens generated
        first_token_time = time.perf_counter()
        prefill_metrics = monitor.end_window("prefill")
        generation_end_time = first_token_time
        monitor.begin_window("generation")

    gen_metrics = monitor.end_window("generation")

    # Count completion tokens via llama-cpp's own tokenizer for accuracy
    full_output = "".join(output_chunks)
    completion_tokens = len(llm.tokenize(full_output.encode())) if full_output else 0

    return EnergyMeasurement(
        pp_tokens=prompt_tokens,
        gen_tokens=completion_tokens,
        pp_time=first_token_time - request_start,
        gen_time=generation_end_time - first_token_time,  # ty:ignore[unsupported-operator]
        pp_j=metrics_to_joules(prefill_metrics),  # type: ignore
        gen_j=metrics_to_joules(gen_metrics),
        idle_power_w=idle_power_w,
    )


def run_llama_cuda_energy_benchmark(quantization: Quantization):
    monitor = ZeusMonitor()

    llm = llama_cpp.Llama(
        quantization.to_ttuf_model_name(),
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=PROMPT_TOKENS + GENERATION_TOKENS,
    )

    for warmup_id in range(WARMUP_RUNS):
        print(f"Warmup run {warmup_id + 1}/{WARMUP_RUNS}...")
        # Count prompt tokens with llama-cpp's own tokenizer
        prompt = generate_random_token_prompt(
            tokenizer=llm.tokenizer(), num_tokens=PROMPT_TOKENS, seed=-1 - warmup_id
        )
        prompt_tokens = len(llm.tokenize(llm.detokenize(prompt)))
        _llama_cuda_single_inference(llm, prompt, prompt_tokens, monitor)

    measurements = []
    for run_id in range(REPEAT_RUNS):
        print(f"Measured run {run_id + 1}/{REPEAT_RUNS}...")
        prompt = generate_random_token_prompt(
            tokenizer=llm.tokenizer(), num_tokens=PROMPT_TOKENS, seed=run_id
        )
        prompt_tokens = len(llm.tokenize(llm.detokenize(prompt)))
        energy_measurement = _llama_cuda_single_inference(
            llm, prompt, prompt_tokens, monitor
        )
        measurements.append(energy_measurement)

    return measurements


# ============================================================================
# CLI Argument Parsing
# ============================================================================


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Energy benchmark for llama-cpp on CUDA"
    )
    parser.add_argument(
        "--quantization",
        type=Quantization,
        choices=list(Quantization),
        default=Quantization.INT4,
        help="Model quantization format",
    )
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=1,
        help="Seconds to measure idle power baseline",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs before measurements",
    )
    parser.add_argument(
        "--repeat-runs", type=int, default=5, help="Number of measured runs to execute"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Number of tokens in the random prompt",
    )
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=128,
        help="Max number of tokens to generate",
    )
    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    IDLE_SECONDS = args.idle_seconds
    REPEAT_RUNS = args.repeat_runs
    WARMUP_RUNS = args.warmup_runs
    PROMPT_TOKENS = args.prompt_tokens
    GENERATION_TOKENS = args.generation_tokens

    measurements = run_llama_cuda_energy_benchmark(args.quantization)

    # Save results to jsonl
    with open(f"llama_cuda_energy_{args.quantization.value}.jsonl", "w") as f:
        for m in measurements:
            f.write(json.dumps(asdict(m)) + "\n")

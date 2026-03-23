# server utils
import argparse
import asyncio
import csv
import statistics
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import aiohttp



@dataclass
class ServerHandle:
    process: subprocess.Popen
    url: str
    pid: int
    server_cmd: str | list[str]
    model: str = ""

async def poll_health(url: str, timeout: float = 60):
    """Block until the server responds 200 at *url*, or raise after *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as s:
                async with s.get(url) as r:
                    if r.status == 200:
                        return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"Server did not become healthy at {url} within {timeout}s")


async def start_server(
    server_cmd: str | list[str],
    model: str,
    port: int | None = None,
) -> ServerHandle:
    """Start the serving framework, wait for health, and return a handle."""

    if isinstance(server_cmd, str):
        server_cmd = server_cmd.split()
    cmd = server_cmd + ["--model", model, "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}/v1/chat/completions"
    health_url = f"http://localhost:{port}/v1/models"
    handle = ServerHandle(process=proc, url=url, pid=proc.pid, server_cmd=server_cmd, model=model)
    await poll_health(health_url)
    return handle

def stop_server(handle: ServerHandle):
    """SIGTERM → wait 5s → SIGKILL."""
    proc = handle.process
    if proc.poll() is not None:
        return
    print("Stopping %s (pid %d) …", handle.server_cmd, handle.pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("Server stopped.")




class Quantization(Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"

    def to_mlx_model_name(self) -> str:
        if self == Quantization.FP16:
            return "mlx-community/Qwen2.5-7B-Instruct-4bit"
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

@dataclass
class BenchmarkResults:
    pp_tps_mean: float
    pp_tps_std: float
    gen_tps_mean: float
    gen_tps_std: float

@dataclass
class FrameworkResults:
    framework: Literal["MLX", "LLAMA_CPP_MPS", "LLAMA_CPP_CUDA"]
    benchmark_results: dict[Quantization, BenchmarkResults]

@dataclass
class InferenceMetrics:
    stream_id: int
    ttft_ms: float
    total_time_ms: float
    output_tokens: int
    success: bool
    error: str | None = None


async def fetch_inference_metrics(api_url: str, payload: dict, stream_id: int) -> InferenceMetrics:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    t_start = time.time()
    t_first_token = None
    token_count = 0

    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()

                async for line in response.content:
                    if not line:
                        continue

                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line == "data: [DONE]":
                        break

                    if decoded_line.startswith("data: "):
                        if t_first_token is None:
                            t_first_token = time.time()

                        token_count += 1

                t_end = time.time()

                return InferenceMetrics(
                    stream_id=stream_id,
                    ttft_ms=(t_first_token - t_start) * 1000 if t_first_token is not None else -1,
                    total_time_ms=(t_end - t_start) * 1000,
                    output_tokens=token_count,
                    success=True
                )

        except Exception as e:
            return InferenceMetrics(
                stream_id=stream_id,
                ttft_ms=-1,
                total_time_ms=-1,
                output_tokens=-1,
                success=False,
                error=str(e)
            )

def build_payload(system_prompt: str, user_prompt: str, max_tokens: int, run_id: int) -> dict:
    prefix = f"[Run {run_id}] "
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": prefix + system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "stream": True
    }

# Llama.cpp benchmark (MPS)
async def run_llama_cpp_mps_benchmark_single_quantization(quantization: Quantization) -> BenchmarkResults:
    # start Llama.cpp server
    model_name = quantization.to_ttuf_model_name()
    server_handle = await start_server(server_cmd="llama-server", model=model_name, port=8080)

    metrics = []
    # warmup
    for i in range(WARMUP_RUNS):
        payload = build_payload(SYSTEM_PROMPT, USER_PROMPT, MAX_TOKENS, run_id=-1 * i)
        print(f"\tWarmup run {i+1}/{WARMUP_RUNS}...")
        await fetch_inference_metrics(server_handle.url, payload, stream_id=-1)

    # benchmark
    for i in range(REPEAT_RUNS):
        print(f"\tBenchmark run {i+1}/{REPEAT_RUNS}...")
        payload = build_payload(SYSTEM_PROMPT, USER_PROMPT, MAX_TOKENS, run_id=i)
        metric = await fetch_inference_metrics(server_handle.url, payload, stream_id=i)
        metrics.append(metric)

    stop_server(server_handle)

    pp_tps_values = [m.output_tokens / (m.ttft_ms / 1000) for m in metrics if m.success and m.ttft_ms > 0]
    gen_tps_values = [m.output_tokens / (m.total_time_ms / 1000) for m in metrics if m.success and m.total_time_ms > 0]
    return BenchmarkResults(
        pp_tps_mean=statistics.mean(pp_tps_values) if pp_tps_values else 0,
        pp_tps_std=statistics.stdev(pp_tps_values) if len(pp_tps_values) > 1 else 0,
        gen_tps_mean=statistics.mean(gen_tps_values) if gen_tps_values else 0,
        gen_tps_std=statistics.stdev(gen_tps_values) if len(gen_tps_values) > 1 else 0
    )

async def run_llama_cpp_benchmark() -> FrameworkResults:
    results = {}
    for quantization in Quantization:
        print(f"Running Llama.cpp (CUDA) benchmark for quantization: {quantization.name}")
        benchmark_result = await run_llama_cpp_mps_benchmark_single_quantization(quantization)
        results[quantization] = benchmark_result
    return FrameworkResults(framework="LLAMA_CPP_CUDA", benchmark_results=results)

if __name__ == "__main__":
    import signal

    parser = argparse.ArgumentParser(description="Benchmark Llama.cpp with different quantizations.")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful AI assistant.", help="System prompt for the benchmark")
    parser.add_argument("--user-prompt", type=str, default="Please tell me about the benefits of Apple Silicon over CUDA for LLM inference.", help="User prompt for the benchmark")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for generation")
    parser.add_argument("--repeat-runs", type=int, default=3, help="Number of benchmark runs to perform")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs to perform before benchmarking")
    args = parser.parse_args()

    SYSTEM_PROMPT = args.system_prompt
    USER_PROMPT = args.user_prompt
    MAX_TOKENS = args.max_tokens
    REPEAT_RUNS = args.repeat_runs
    WARMUP_RUNS = args.warmup_runs

    framework_results = asyncio.run(run_llama_cpp_benchmark())

    # dump framework results to CSV

    with open("framework_results.csv", "w", newline="") as csvfile:
        fieldnames = ["quantization", "pp_tps_mean", "pp_tps_std", "gen_tps_mean", "gen_tps_std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for quantization, benchmark_result in framework_results.benchmark_results.items():
            writer.writerow({
                "quantization": quantization.name,
                "pp_tps_mean": benchmark_result.pp_tps_mean,
                "pp_tps_std": benchmark_result.pp_tps_std,
                "gen_tps_mean": benchmark_result.gen_tps_mean,
                "gen_tps_std": benchmark_result.gen_tps_std
            })

from sys import stderr
import sys
import asyncio
import json
import os
import signal
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

import aiohttp
from faker import Faker
from transformers import AutoTokenizer

OLLAMA_DEFAULT_PORT = 11434
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
assert TOKENIZER is not None


@dataclass
class ServerHandle:
    process: asyncio.subprocess.Process
    url: str
    pid: int
    server_cmd: str | list[str]
    model: str = ""


@dataclass
class BenchmarkResults:
    pp_tps_mean: float
    pp_tps_std: float
    gen_tps_mean: float
    gen_tps_std: float
    batch_size: int
    pp_tokens: int
    gen_tokens: int

    @property
    def time_to_first_token(self) -> float:
        return self.pp_tokens * self.batch_size / self.pp_tps_mean

    @property
    def time_per_output_token(self) -> float:
        return self.gen_tokens * self.batch_size / self.gen_tps_mean / self.gen_tokens


class Quantization(Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"

    def to_mlx_model_name(self, model_size: Literal["7B", "14B"] = "7B") -> str:
        if self == Quantization.FP16:
            return f"mlx-community/Qwen2.5-{model_size}-Instruct-bf16"
        elif self == Quantization.INT8:
            return f"mlx-community/Qwen2.5-{model_size}-Instruct-8bit"
        elif self == Quantization.INT4:
            return f"mlx-community/Qwen2.5-{model_size}-Instruct-4bit"
        else:
            raise ValueError(f"Unsupported quantization: {self}")

    def to_ttuf_model_name(self, model_size: Literal["7B", "14B"] = "7B") -> str:
        if self == Quantization.FP16:
            return f"models/qwen2.5-{model_size}/Qwen2.5-{model_size}-Instruct-f16.gguf"
        elif self == Quantization.INT8:
            return (
                f"models/qwen2.5-{model_size}/Qwen2.5-{model_size}-Instruct-Q8_0.gguf"
            )
        elif self == Quantization.INT4:
            return (
                f"models/qwen2.5-{model_size}/Qwen2.5-{model_size}-Instruct-Q4_K_M.gguf"
            )
        else:
            raise ValueError(f"Unsupported quantization: {self}")

    def to_ollama_model_name(self, model_size: Literal["7B", "14B"] = "7B") -> str:
        if self == Quantization.FP16:
            return f"qwen2.5:{model_size.lower()}-instruct-fp16"
        elif self == Quantization.INT8:
            return f"qwen2.5:{model_size.lower()}-instruct-q8_0"
        elif self == Quantization.INT4:
            return f"qwen2.5:{model_size.lower()}-instruct-q4_K_M"
        else:
            raise ValueError(f"Unsupported quantization: {self}")


async def poll_health(url: str, timeout: float = 20):
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
    server_cmd: str | list[str], model: str, port: int, **env: str
) -> ServerHandle:
    """Start the serving framework asynchronously, wait for health, and return a handle."""
    if isinstance(server_cmd, str):
        server_cmd = server_cmd.split()

    proc = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={**os.environ.copy(), **env},
    )

    url = f"http://localhost:{port}/v1/chat/completions"
    health_url = f"http://localhost:{port}/v1/models"
    handle = ServerHandle(
        process=proc, url=url, pid=proc.pid, server_cmd=server_cmd, model=model
    )
    await poll_health(health_url)
    return handle


async def stop_server(handle: ServerHandle):
    """SIGTERM → wait 5s → SIGKILL."""
    proc = handle.process
    if proc.returncode is not None:
        return
    print(f"Stopping {handle.server_cmd} (pid {handle.pid}) ...", file=sys.stderr)
    proc.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    print("Server stopped.", file=sys.stderr)


@asynccontextmanager
async def run_server(server_cmd: str | list[str], model: str, port: int, **env: str):
    """Context manager for start_server + stop_server."""
    handle = await start_server(server_cmd, model, port, **env)
    try:
        yield handle
    finally:
        await stop_server(handle)


@dataclass
class RequestResult:
    ttft_ms: float = float("nan")
    generation_time_s: float = float("nan")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    content: str = ""
    error: str | None = None


async def streaming_request(
    url: str, payload: dict, timeout_s: float = 600
) -> RequestResult:
    if "stream_options" not in payload:
        payload["stream_options"] = {"include_usage": True}

    result = RequestResult()
    t_start = time.perf_counter()
    t_first = None

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:500]}"
                    return result

                async for line in resp.content:
                    if not line:
                        continue

                    decoded_line = line.decode("utf-8", errors="replace")
                    if decoded_line == "data: [DONE]":
                        break

                    if not decoded_line.startswith("data: "):
                        continue

                    data_str = decoded_line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract exact hardware token counts from the final usage chunk
                    if "usage" in data and data["usage"] is not None:
                        result.prompt_tokens = data["usage"].get("prompt_tokens", 0)
                        result.completion_tokens = data["usage"].get(
                            "completion_tokens", result.completion_tokens
                        )

                    # Extract content if present
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token_text = delta.get("content")
                        if token_text:
                            # Capture TTFT on the first valid JSON payload
                            if t_first is None:
                                t_first = time.perf_counter()
                                result.ttft_ms = (t_first - t_start) * 1000
                            result.content += token_text

    except Exception as exc:
        result.error = str(exc)
        return result

    t_end = time.perf_counter()
    if t_first is not None:
        result.generation_time_s = t_end - t_first
    return result


def build_payload(*history: dict[str, str], max_tokens: int, model: str) -> dict:
    messages = list(history)
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "model": model,
    }


def generate_random_token_prompt(
    tokenizer, num_tokens: int, seed: int | None = None
) -> list[int]:
    fake = Faker()
    if seed is not None:
        fake.seed_instance(seed)
    text_buffer = ""
    while len(text_buffer.split()) < 2 * num_tokens:
        text_buffer += fake.sentence() + " "
    return tokenizer.encode(text_buffer)[:num_tokens]


def count_tokens_in_messages(messages: list[dict]) -> int:
    """Count tokens for a messages list using apply_chat_template."""
    tok = TOKENIZER
    rendered = tok.apply_chat_template(  # ty:ignore[unresolved-attribute]
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tok.encode(rendered))  # ty:ignore[unresolved-attribute]


async def run_warmup(handle: ServerHandle):
    system_prompt = TOKENIZER.decode(  # ty:ignore[unresolved-attribute]
        generate_random_token_prompt(
            tokenizer=TOKENIZER, num_tokens=WARMUP_SYSTEM_PROMPT_TOKENS, seed=42
        )
    )
    user_prompt = TOKENIZER.decode(  # ty:ignore[unresolved-attribute]
        generate_random_token_prompt(
            tokenizer=TOKENIZER, num_tokens=WARMUP_INPUT_TOKENS, seed=43
        )
    )
    payload = build_payload(
        {"role": "system", "content": system_prompt},  # ty:ignore[invalid-argument-type]
        {"role": "user", "content": user_prompt},  # ty:ignore[invalid-argument-type]
        max_tokens=WARMUP_OUTPUT_TOKENS,
        model=handle.model,
    )
    r = await streaming_request(handle.url, payload)
    if r.error:
        raise RuntimeError(f"Warm-up failed: {r.error}")


def generate_controlled_prompt(
    tokenizer, run_id: int, cycle: int, agent_id: str, target_tokens: int
) -> str:
    prefix_text = f"UID:{run_id}-{cycle}-{agent_id}\n"
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    suffix_text = "\nTask: Count incrementally from 1 to 100,000. Do not stop. Do not provide commentary. Begin now: 1, 2, 3, 4, 5, 6, 7,"
    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)

    required_padding = target_tokens - len(prefix_tokens) - len(suffix_tokens)
    if required_padding < 0:
        raise ValueError(
            f"Target token count ({target_tokens}) is insufficient to hold the control blocks."
        )

    padding_base_tokens = tokenizer.encode(
        "system diagnostic routine running ", add_special_tokens=False
    )
    padding_tokens = []
    while len(padding_tokens) < required_padding:
        padding_tokens.extend(padding_base_tokens)
    padding_tokens = padding_tokens[:required_padding]

    final_tokens = prefix_tokens + padding_tokens + suffix_tokens
    assert len(final_tokens) == target_tokens
    return tokenizer.decode(final_tokens)


async def evaluate_framework_throughput(handle: ServerHandle) -> BenchmarkResults:
    print("Running warm-up cycles...", file=sys.stderr)
    await run_warmup(handle)

    system_prompt = "You are a counting algorithm. You do not output text. You only output sequential integers, one per line."

    user_msgs = [
        generate_controlled_prompt(
            tokenizer=TOKENIZER, run_id=run, cycle=0, agent_id="0", target_tokens=1024
        )
        for run in range(REPEAT_RUNS)
    ]

    results = []

    n_tokens = []
    for message in user_msgs:
        payload = build_payload(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
            max_tokens=GENERATION_TOKENS,
            model=handle.model,
        )
        r = await streaming_request(handle.url, payload)
        if r.error:
            raise RuntimeError(f"Warm-up message failed: {r.error}")

        pp_tps = r.prompt_tokens / (r.ttft_ms / 1000) if r.ttft_ms > 0 else float("nan")
        gen_tps = (
            r.completion_tokens / r.generation_time_s
            if r.generation_time_s > 0
            else float("nan")
        )
        results.append((pp_tps, gen_tps))
        n_tokens.append((r.prompt_tokens, r.completion_tokens))

    assert all(n == n_tokens[0] for n in n_tokens), (
        f"Token counts varied across runs: {n_tokens}"
    )

    return BenchmarkResults(
        pp_tps_mean=statistics.mean(pp for pp, _ in results),
        pp_tps_std=statistics.stdev(pp for pp, _ in results)
        if len(results) > 1
        else 0.0,
        gen_tps_mean=statistics.mean(gen for _, gen in results),
        gen_tps_std=statistics.stdev(gen for _, gen in results)
        if len(results) > 1
        else 0.0,
        pp_tokens=n_tokens[0][0],
        gen_tokens=n_tokens[0][1],
        batch_size=1,
    )


async def evaluate_ollama_cuda_throughput(
    quantization: Quantization,
) -> BenchmarkResults:
    # env = {"OLLAMA_BACKEND": "mps"}
    async with run_server(
        "ollama serve",
        model=quantization.to_ollama_model_name(),
        port=OLLAMA_DEFAULT_PORT,
    ) as handle:
        return await evaluate_framework_throughput(handle)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Ollama CUDA performance with different quantizations."
    )
    parser.add_argument(
        "--quantization",
        type=Quantization,
        choices=list(Quantization),
        default=Quantization.INT4,
        help="Quantization format to test.",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=3,
        help="Number of times to repeat the benchmark for averaging.",
    )
    parser.add_argument(
        "--warmup-system-prompt-tokens",
        type=int,
        default=512,
        help="Number of tokens in the warm-up system prompt.",
    )
    parser.add_argument(
        "--warmup-input-tokens",
        type=int,
        default=512,
        help="Number of tokens in the warm-up user prompt.",
    )
    parser.add_argument(
        "--warmup-output-tokens",
        type=int,
        default=512,
        help="Number of tokens to generate during warm-up.",
    )
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=1024,
        help="Number of tokens to generate during the benchmark.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    QUANTIZATION = args.quantization
    REPEAT_RUNS = args.repeat_runs
    WARMUP_SYSTEM_PROMPT_TOKENS = args.warmup_system_prompt_tokens
    WARMUP_INPUT_TOKENS = args.warmup_input_tokens
    WARMUP_OUTPUT_TOKENS = args.warmup_output_tokens
    GENERATION_TOKENS = args.generation_tokens

    results = asyncio.run(evaluate_ollama_cuda_throughput(QUANTIZATION))
    json.dump(asdict(results), fp=sys.stdout)

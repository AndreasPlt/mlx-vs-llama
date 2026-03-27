import argparse
import asyncio
import json
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Self

import aiohttp
import psutil
import pynvml
from faker import Faker
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# ── Type aliases ──────────────────────────────────────────────────────────────

type Agent = str
type Cycle = int

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ServerHandle:
    process: asyncio.subprocess.Process
    url: str
    pid: int
    server_cmd: str | list[str]
    model: str = ""


@dataclass
class RequestResult:
    ttft_ms: float = float("nan")
    generation_time_s: float = float("nan")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    content: str = ""
    error: str | None = None


@dataclass
class ConversationTurnResult:
    ttft_s: float
    gen_tps: float
    pp_token: int
    gen_tokens: int
    peak_vram_mb: float
    host_ram_mb: int
    disk_read_mb: int
    disk_write_mb: int
    swap_used_mb: int

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)


@dataclass
class ConversationCycleResult:
    agent_results: dict[Agent, ConversationTurnResult]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        agent_results = {
            agent: ConversationTurnResult.from_dict(res)
            for agent, res in data["agent_results"].items()
        }
        return cls(agent_results=agent_results)


@dataclass
class ConversationRunResult:
    cycle_results: list[ConversationCycleResult]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        cycle_results = [
            ConversationCycleResult.from_dict(cycle) for cycle in data["cycle_results"]
        ]
        return cls(cycle_results=cycle_results)


# ── Server management ─────────────────────────────────────────────────────────


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

    env_variables = os.environ.copy()
    env_variables.update(env)

    proc = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=env_variables,
    )

    url = f"http://localhost:{port}/v1/chat/completions"
    health_url = f"http://localhost:{port}/v1/models"
    handle = ServerHandle(
        process=proc, url=url, pid=proc.pid, server_cmd=server_cmd, model=model
    )
    await poll_health(health_url)
    return handle


async def stop_server(handle: ServerHandle):
    """SIGTERM → wait 5s asynchronously → SIGKILL."""
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


# ── Request handling ──────────────────────────────────────────────────────────


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

                    # Capture TTFT on the first valid JSON payload
                    if t_first is None:
                        t_first = time.perf_counter()
                        result.ttft_ms = (t_first - t_start) * 1000

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
                            result.content += token_text

    except Exception as exc:
        result.error = str(exc)
        return result

    t_end = time.perf_counter()
    if t_first is not None:
        result.generation_time_s = t_end - t_first
    return result


# ── Prompt utilities ──────────────────────────────────────────────────────────


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


def generate_controlled_prompt(
    tokenizer, run_id: int, cycle: int, agent_id: str, target_tokens: int
) -> str:
    # 1. Cache Breaker
    prefix_text = f"UID:{run_id}-{cycle}-{agent_id}\n"
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    # 3. Execution Lock
    suffix_text = "\nTask: Count incrementally from 1 to 100,000. Do not stop. Do not provide commentary. Begin now: 1, 2, 3, 4, 5, 6, 7,"
    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)

    # 2. Padding Block Calculation
    required_padding = target_tokens - len(prefix_tokens) - len(suffix_tokens)
    if required_padding < 0:
        raise ValueError(
            f"Target token count ({target_tokens}) is insufficient to hold the control blocks."
        )

    padding_base_text = "system diagnostic routine running "
    padding_base_tokens = tokenizer.encode(padding_base_text, add_special_tokens=False)

    padding_tokens = []
    while len(padding_tokens) < required_padding:
        padding_tokens.extend(padding_base_tokens)

    padding_tokens = padding_tokens[:required_padding]

    # Assembly
    final_tokens = prefix_tokens + padding_tokens + suffix_tokens
    assert len(final_tokens) == target_tokens, "Token length mismatch during assembly."

    return tokenizer.decode(final_tokens)


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


def count_tokens_in_messages(messages: list[dict]) -> int:
    """Count tokens for a messages list using apply_chat_template."""
    tok = TOKENIZER
    rendered = tok.apply_chat_template(  # ty:ignore[unresolved-attribute]
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tok.encode(rendered))  # ty:ignore[unresolved-attribute]


# ── Benchmark logic ───────────────────────────────────────────────────────────


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


async def run_multi_conversation_run(
    handle: ServerHandle, run_id: int
) -> ConversationRunResult:

    print("Running warm-up cycles...", file=sys.stderr)
    await run_warmup(handle)

    system_prompt = "You are a counting algorithm. You do not output text. You only output sequential integers, one per line."

    user_msgs = {
        (cycle, agent): generate_controlled_prompt(
            tokenizer=TOKENIZER,
            run_id=run_id,
            cycle=cycle,
            agent_id=agent,
            target_tokens=USER_MSG_TOKENS,
        )
        for cycle in range(1, NUM_CYCLES + 1)
        for agent in AGENTS
    }

    # Initialize process monitor
    try:
        server_process = psutil.Process(handle.pid)
    except psutil.NoSuchProcess:
        raise RuntimeError(f"Server PID {handle.pid} not found.")

    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
        int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    )  # Assuming single GPU for simplicity
    histories: dict[str, list[dict]] = {
        agent: [{"role": "system", "content": system_prompt}] for agent in AGENTS
    }
    results = []

    for cycle in range(1, NUM_CYCLES + 1):
        cycle_results: dict[str, ConversationTurnResult] = {}
        for agent in AGENTS:
            user_text = user_msgs[(cycle, agent)]
            messages = histories[agent]
            messages.append({"role": "user", "content": user_text})

            history_tokens = count_tokens_in_messages(messages)

            payload = build_payload(
                *messages, max_tokens=MAX_CYCLE_TOKENS, model=handle.model
            )

            mem_info_before = server_process.memory_info()
            vram_info_before = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            # have to rely on global disk IO access as MacOS doesn't provide per-process disk IO counters
            disk_io_before = psutil.disk_io_counters()

            r = await streaming_request(handle.url, payload)

            mem_info_after = server_process.memory_info()
            # have to rely on global disk IO access as MacOS doesn't provide per-process disk IO counters
            vram_info_after = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            disk_io_after = psutil.disk_io_counters()
            swap_info = psutil.swap_memory()

            rss_mb = int(max(mem_info_before.rss, mem_info_after.rss) / (1024 * 1024))
            disk_read_mb = int(
                (disk_io_after.read_bytes - disk_io_before.read_bytes) / (1024 * 1024)
            )
            vram_peak_mb = int(
                max(vram_info_before.used, vram_info_after.used) / (1024 * 1024)
            )
            disk_write_mb = int(
                (disk_io_after.write_bytes - disk_io_before.write_bytes) / (1024 * 1024)
            )
            swap_used_mb = int(swap_info.used / (1024 * 1024))

            if r.error:
                print(
                    f"  ERROR cycle={cycle} agent={agent}: {r.error}", file=sys.stderr
                )
                # Append empty assistant turn so history is consistent
                histories[agent].append({"role": "assistant", "content": ""})
                continue

            ttft_s = r.ttft_ms / 1000
            gen_tps = (
                r.completion_tokens / r.generation_time_s
                if r.generation_time_s > 0
                else float("nan")
            )

            # Update history with this turn
            messages.append({"role": "assistant", "content": r.content})
            cycle_results[agent] = ConversationTurnResult(
                ttft_s=ttft_s,
                gen_tps=gen_tps,
                pp_token=history_tokens,
                gen_tokens=r.completion_tokens,
                host_ram_mb=rss_mb,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                swap_used_mb=swap_used_mb,
                peak_vram_mb=vram_peak_mb,
            )

        results.append(ConversationCycleResult(agent_results=cycle_results))

    return ConversationRunResult(cycle_results=results)


async def run_ollama_cuda_multi_conversation_benchmark() -> list[ConversationRunResult]:
    OLLAMA_DEFAULT_PORT = 11434

    results = []

    ollama_env = {
        "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
        "OLLAMA_NUM_PARALLEL": str(NUM_PARALLEL),
    }

    for run_id in range(2):
        async with run_server(
            server_cmd="ollama serve",
            model="qwen2.5:14b-instruct-q4_K_M",
            port=OLLAMA_DEFAULT_PORT,
            **ollama_env,
        ) as handle:
            print(
                f"Running multi-conversation Ollama benchmark against server at {handle.url} ...", file=sys.stderr
            )
            ollama_multi_conv_results = await run_multi_conversation_run(
                handle, run_id=run_id
            )
            results.append(ollama_multi_conv_results)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-conversation benchmark against Ollama or similar local serving framework."
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=1,
        help="Number of times to repeat the entire multi-conversation benchmark run (including server restarts).",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=4,
        help="Number of conversation cycles to run per agent.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="List of agent identifiers to simulate in the multi-conversation benchmark.",
    )
    parser.add_argument(
        "--user-msg-tokens",
        type=int,
        default=2048,
        help="Number of tokens in each user message prompt to control for token count impact on performance.",
    )
    parser.add_argument(
        "--max-cycle-tokens",
        type=int,
        default=512,
        help="Maximum total tokens (prompt + generation) for each conversation turn in the benchmark.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=16384,
        help="Context length for the model.",
    )
    parser.add_argument(
        "--num-parallel-conversations",
        type=int,
        default=0,
        help="Number of parallel conversations to simulate. If < 1, it will be set to the number of agents."
    )
    parser.add_argument(
        "--warmup-system-prompt-tokens",
        type=int,
        default=64,
        help="Number of tokens in the system prompt used for warm-up before the main benchmark runs.",
    )
    parser.add_argument(
        "--warmup-input-tokens",
        type=int,
        default=64,
        help="Number of tokens in the user prompt used for warm-up before the main benchmark runs.",
    )
    parser.add_argument(
        "--warmup-output-tokens",
        type=int,
        default=16,
        help="Number of tokens to request in the generation during warm-up before the main benchmark runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    REPEAT_RUNS = args.repeat_runs
    NUM_CYCLES = args.num_cycles
    AGENTS = args.agents
    USER_MSG_TOKENS = args.user_msg_tokens
    MAX_CYCLE_TOKENS = args.max_cycle_tokens
    NUM_PARALLEL = args.num_parallel_conversations if args.num_parallel_conversations > 0 else len(AGENTS)
    CONTEXT_LENGTH = args.context_length
    WARMUP_SYSTEM_PROMPT_TOKENS = args.warmup_system_prompt_tokens
    WARMUP_INPUT_TOKENS = args.warmup_input_tokens
    WARMUP_OUTPUT_TOKENS = args.warmup_output_tokens

    final_results = asyncio.run(run_ollama_cuda_multi_conversation_benchmark())

    # Output results as JSONL
    for run_results in final_results:
        json_line = json.dumps(asdict(run_results))
        print(json_line)

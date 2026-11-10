"""Smoke test of a running proxy with the official openai SDK (real Hoplite runs: costs a little).

    set PROXY_API_KEY=sk-hop-...            (PowerShell: $env:PROXY_API_KEY = "sk-hop-...")
    python scripts/smoke_openai.py [--base-url http://127.0.0.1:8000/v1] [--model hoplite-default]

Checks: model list, a non-streaming answer, a streamed follow-up that must land in the same Hoplite
thread, and a client tool round trip (tool_calls -> tool result -> final answer) through the tool bridge.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import openai

FIRST = "Reply with exactly the word: pong. Do not run tools or touch any files."
SECOND = "Now reply with exactly the two words: pong again. Do not run tools or touch any files."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="hoplite-default")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    api_key = os.environ.get("PROXY_API_KEY", "")
    if not api_key.startswith("sk-hop-"):
        print("error: set PROXY_API_KEY to a key issued by the proxy (sk-hop-...)", file=sys.stderr)
        return 2
    client = openai.OpenAI(base_url=args.base_url, api_key=api_key, max_retries=0, timeout=args.timeout)
    failures = 0

    models = [m.id for m in client.models.list().data]
    print(f"[models] {len(models)}: {', '.join(models[:8])}{' ...' if len(models) > 8 else ''}")

    started = time.monotonic()
    raw = client.chat.completions.with_raw_response.create(
        model=args.model, messages=[{"role": "user", "content": FIRST}]
    )
    first = raw.parse()
    thread_id = raw.headers.get("x-hoplite-thread-id")
    reply = first.choices[0].message.content or ""
    print(f"[non-stream] {time.monotonic() - started:.1f}s thread={thread_id} model={first.model} "
          f"finish={first.choices[0].finish_reason} usage={first.usage}")
    print(f"             {reply!r}")

    started = time.monotonic()
    raw_stream = client.chat.completions.with_raw_response.create(
        model=args.model, stream=True, stream_options={"include_usage": True},
        messages=[{"role": "user", "content": FIRST}, {"role": "assistant", "content": reply},
                  {"role": "user", "content": SECOND}],
    )
    follow_thread = raw_stream.headers.get("x-hoplite-thread-id")
    print("[stream]     ", end="", flush=True)
    usage = None
    chunks = 0
    for chunk in raw_stream.parse():
        if chunk.choices and chunk.choices[0].delta.content:
            chunks += 1
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:
            usage = chunk.usage
    print(f"\n             {time.monotonic() - started:.1f}s chunks={chunks} usage={usage}")
    if follow_thread != thread_id:
        failures += 1
        print(f"FAIL: follow-up went to thread {follow_thread}, expected {thread_id}")
    else:
        print(f"[thread]     follow-up continued thread {thread_id}")

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city, measured by the client.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    conversation = [{"role": "user", "content": "What is the weather in Kyiv right now? Use the get_weather tool."}]
    started = time.monotonic()
    first_tool = client.chat.completions.create(model=args.model, messages=conversation, tools=tools)
    choice = first_tool.choices[0]
    if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
        failures += 1
        print(f"FAIL: expected a tool call, got finish={choice.finish_reason} {choice.message.content!r}")
    else:
        call = choice.message.tool_calls[0]
        print(f"[tools]      {time.monotonic() - started:.1f}s -> {call.function.name}({call.function.arguments})")
        conversation += [choice.message.model_dump(exclude_none=True),
                         {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 17, "sky": "sunny"}'}]
        started = time.monotonic()
        answer = client.chat.completions.create(model=args.model, messages=conversation, tools=tools)
        text = answer.choices[0].message.content or ""
        print(f"[tools]      {time.monotonic() - started:.1f}s answer: {text!r}")
        if "17" not in text:
            failures += 1
            print("FAIL: the answer does not use the tool result")

    print("PASS" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

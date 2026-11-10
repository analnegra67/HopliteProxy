"""Smoke test of a running proxy with the official openai SDK (real Hoplite runs: costs a little).

    set PROXY_API_KEY=sk-hop-...            (PowerShell: $env:PROXY_API_KEY = "sk-hop-...")
    python scripts/smoke_openai.py [--base-url http://127.0.0.1:8000/v1] [--model hoplite-default]

Checks: model list, a non-streaming answer, a streamed follow-up that must land in the same Hoplite
thread, and that `tools` is rejected with 400.
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

    try:
        client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
        )
        failures += 1
        print("FAIL: a request with tools was accepted")
    except openai.BadRequestError as exc:
        print(f"[tools]      rejected as expected: {exc.code}")

    print("PASS" if failures == 0 else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

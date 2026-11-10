"""Live probe of the Hoplite API. Reads HOPLITE_API_KEY (and HOPLITE_PROJECT_ID) from the environment or .env.

    python scripts/probe_hoplite.py list
        Read-only: the key's capabilities, accessible projects and available models.

    python scripts/probe_hoplite.py run [--project-id proj_...] [--prompt "..."] [--model ...]
        Creates ONE thread (spend limit $0.25 by default, autoMerge off), records every raw SSE frame
        of its event stream to scripts/out/probe-<timestamp>.txt and prints a summary of the frame
        types, in particular the draft frames whose layout Hoplite does not document.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_PROMPT = (
    "Reply in plain text with three short sentences about why the sky is blue. "
    "Do not run tools, do not read or modify any files."
)
TERMINAL = {"completed", "failed", "cancelled", "waiting"}


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    hoplite_api_key: str = ""
    hoplite_project_id: str = ""
    hoplite_base_url: str = "https://api.hoplite.sh"


async def get(http: httpx.AsyncClient, path: str, **params: Any) -> dict[str, Any]:
    resp = await http.get(path, params={k: v for k, v in params.items() if v is not None})
    if resp.status_code >= 400:
        raise RuntimeError(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


async def cmd_list(http: httpx.AsyncClient) -> None:
    try:
        caps = await get(http, "/api/capabilities")
        print(f"principal: {caps.get('principalId')}  org: {caps.get('orgId')}")
        print(f"permissions: {caps.get('permissions')}")
        print(f"project allowlist: {caps.get('projectIds')}")
    except RuntimeError as exc:
        print(f"capabilities: {exc}")
    try:
        projects = (await get(http, "/api/projects", limit=100)).get("projects") or []
        print(f"\nprojects ({len(projects)}):")
        for p in projects:
            repos = [r.get("repoFullName") for r in p.get("repos") or []]
            print(f"  {p['id']:<28} {p.get('name')!r:<30} repos={repos} defaultModel={p.get('defaultModel')}")
    except RuntimeError as exc:
        print(f"projects: {exc}")
    try:
        models = await get(http, "/api/model-providers")
        print(f"\ndefault model: {(models.get('runConfig') or {}).get('defaultModelId')}")
        for m in models.get("models") or []:
            print(f"  {m['id']:<32} {str(m.get('provider')):<12} {m.get('name')}")
    except RuntimeError as exc:
        print(f"models: {exc}")


def _parse_frames(raw_lines: list[str]) -> list[dict[str, Any]]:
    frames, current = [], {"event": None, "id": None, "data": []}
    for line in raw_lines:
        if line == "":
            if current["data"] or current["event"]:
                frames.append({**current, "data": "\n".join(current["data"])})
            current = {"event": None, "id": None, "data": []}
        elif line.startswith(":"):
            frames.append({"event": "(comment)", "id": None, "data": line})
        else:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data":
                current["data"].append(value)
            elif field in ("event", "id"):
                current[field] = value
    return frames


async def cmd_run(http: httpx.AsyncClient, env: Env, args: argparse.Namespace) -> int:
    project_id = args.project_id or env.hoplite_project_id
    if not project_id:
        print("error: pass --project-id or set HOPLITE_PROJECT_ID (see: probe_hoplite.py list)", file=sys.stderr)
        return 2
    op = f"probe-{uuid.uuid4().hex[:16]}"
    body: dict[str, Any] = {
        "projectId": project_id, "prompt": args.prompt, "autoMerge": False,
        "spendLimitMicros": args.spend_limit_micros, "metadata": {"source": "openai-proxy-probe"},
    }
    if args.model:
        body["model"] = args.model
    started = time.monotonic()
    resp = await http.post("/api/threads", json=body, headers={"Idempotency-Key": f"{op}:thread"})
    print(f"POST /api/threads -> HTTP {resp.status_code}")
    if resp.status_code >= 400:
        print(resp.text[:1000])
        return 1
    created = resp.json()
    thread_id = created["thread"]["id"]
    run_id = (created.get("run") or {}).get("id") or created["thread"].get("latestRunId")
    print(f"thread {thread_id}  run {run_id}  model {created['thread'].get('modelId')}")
    cursor = (await get(http, "/api/events/head", threadId=thread_id)).get("cursor")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"probe-{time.strftime('%Y%m%d-%H%M%S')}.txt"
    raw_lines: list[str] = []
    settled_status = None
    deadline = time.monotonic() + args.timeout
    with out_path.open("w", encoding="utf-8") as out:
        out.write(f"# thread {thread_id} run {run_id}\n# create response: {json.dumps(created)[:4000]}\n\n")
        while settled_status is None and time.monotonic() < deadline:
            headers = {"Accept": "text/event-stream"}
            if cursor:
                headers["Last-Event-ID"] = cursor
            try:
                async with http.stream("GET", "/api/events/stream", params={"threadId": thread_id, "cursor": cursor},
                                       headers=headers, timeout=httpx.Timeout(10, read=45)) as stream:
                    print(f"GET /api/events/stream -> HTTP {stream.status_code} {stream.headers.get('content-type')}")
                    if stream.status_code != 200:
                        print((await stream.aread()).decode()[:500])
                        break
                    async for line in stream.aiter_lines():
                        out.write(line + "\n")
                        raw_lines.append(line)
                        if line.startswith("id:"):
                            cursor = line[3:].strip()
                        if line.startswith("data:") and time.monotonic() - started < 2 * args.timeout:
                            try:
                                data = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue
                            inner = data.get("data") if isinstance(data.get("data"), dict) else {}
                            if str(data.get("type", "")).startswith("run.") and inner.get("status") in TERMINAL:
                                if not run_id or inner.get("id") == run_id:
                                    settled_status = inner.get("status")
                                    break
                        if time.monotonic() > deadline:
                            break
            except httpx.HTTPError as exc:
                print(f"stream interrupted: {exc!r}; reconnecting")
                await asyncio.sleep(1)

        if settled_status is None and run_id:
            print("timeout: stopping the run")
            await http.post(f"/api/threads/{thread_id}/stop", json={"runId": run_id},
                            headers={"Idempotency-Key": f"{op}:stop"})
        state = await get(http, f"/api/threads/{thread_id}/run-state", runId=run_id)
        messages = await get(http, f"/api/threads/{thread_id}/messages", limit=50)
        usage = await get(http, f"/api/threads/{thread_id}/usage")
        out.write(f"\n# run-state: {json.dumps(state)}\n# messages: {json.dumps(messages)[:20000]}\n"
                  f"# usage: {json.dumps(usage)}\n")
        if settled_status == "waiting":
            out.write(f"# questions: {json.dumps(await get(http, f'/api/threads/{thread_id}/questions'))}\n")

    frames = _parse_frames(raw_lines)
    print(f"\nsettled: {settled_status} after {time.monotonic() - started:.1f}s; {len(frames)} frames -> {out_path}")
    print("frame types:", dict(Counter(f["event"] or "(none)" for f in frames)))
    drafts = [f for f in frames if f["event"] and "draft" in f["event"]]
    for frame in (drafts[:2] + drafts[-1:]) if drafts else []:
        print(f"\n--- {frame['event']} (id={frame['id']}) ---\n{frame['data'][:600]}")
    run = state.get("run") or {}
    final = str((run.get("assistantMessage") or {}).get("content"))[:300]
    print(f"\nrun status: {run.get('status')}; final message: {final!r}")
    print(f"usage: in={usage.get('usage', {}).get('inputTokens')} out={usage.get('usage', {}).get('outputTokens')} "
          f"cost_micros={usage.get('usage', {}).get('totalCostMicros')}")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="read-only: capabilities, projects, models")
    run = sub.add_parser("run", help="create one thread and record its event stream")
    run.add_argument("--project-id")
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--model")
    run.add_argument("--spend-limit-micros", type=int, default=250_000)
    run.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    env = Env()
    if not env.hoplite_api_key.startswith("hop_"):
        print("error: set HOPLITE_API_KEY (environment or .env) to a key starting with hop_", file=sys.stderr)
        return 2
    async with httpx.AsyncClient(base_url=env.hoplite_base_url, timeout=30,
                                 headers={"X-Api-Key": env.hoplite_api_key, "Accept": "application/json"}) as http:
        if args.command == "list":
            await cmd_list(http)
            return 0
        return await cmd_run(http, env, args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

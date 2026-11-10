"""Tool bridge: client-side tools (opencode and similar agents) served by a Hoplite agent."""

from __future__ import annotations

import json
import re

import pytest

from tests.conftest import parse_sse
from tests.fake_hoplite import Say, Tool

pytestmark = pytest.mark.anyio

READ = {"type": "function", "function": {
    "name": "read", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}
EDIT = {"type": "function", "function": {
    "name": "edit", "description": "Replace text in a file",
    "parameters": {"type": "object", "required": ["filePath", "oldString", "newString"],
                   "properties": {"filePath": {"type": "string"}, "oldString": {"type": "string"},
                                  "newString": {"type": "string"}}}}}
TOOLS = [READ, EDIT]


def block(*calls: tuple[str, dict]) -> str:
    return "<tool_calls>\n" + json.dumps([{"name": n, "arguments": a} for n, a in calls]) + "\n</tool_calls>"


def tool_result(text: str) -> tuple[str, str, str] | None:
    match = re.search(r'<tool_result id="([^"]*)" name="([^"]*)">\n(.*?)\n</tool_result>', text, re.S)
    return match.groups() if match else None  # type: ignore[return-value]


def user(text: str) -> dict:
    return {"role": "user", "content": text}


async def chat(client, auth, messages, **extra):
    extra.setdefault("tools", TOOLS)
    return await client.post("/v1/chat/completions", headers=auth,
                             json={"model": "gpt-6-sol", "messages": messages, **extra})


def appended_messages(fake) -> list[str]:
    return [c[2]["content"] for c in fake.calls if c[0] == "POST" and c[1].endswith("/messages")]


async def test_tool_call_response(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Say("I will read the file.\n" + block(("read", {"filePath": "a.py"})))]
    resp = await chat(client, auth, [{"role": "system", "content": "You are opencode."}, user("fix a.py")])
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I will read the file."
    [call] = choice["message"]["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_") and call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"filePath": "a.py"}

    prompt = fake.thread_creations()[0]["prompt"]
    assert "<client_tools>" in prompt and '"name": "edit"' in prompt
    assert "<client_instructions>\nThe client agent's own instructions." in prompt
    assert "\n\nYou are opencode.\n</client_instructions>" in prompt
    assert (prompt.index("<client_instructions>\n") < prompt.index("<client_tools>\n")
            < prompt.index("<protocol>\nYou are the reasoning engine"))
    assert "User request:\n\nfix a.py\n\n(Reminder:" in prompt


async def test_streaming_hides_the_protocol(client, auth, fake) -> None:
    reply = "Checking both.\n" + block(("read", {"filePath": "a.py"}), ("read", {"filePath": "b.py"}))
    fake.behaviour = lambda text: [Say(reply, chunks=9)]
    resp = await chat(client, auth, [user("x")], stream=True)
    chunks = parse_sse(resp.text)
    assert chunks[-1] == "[DONE]"
    chunks = chunks[:-1]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert "<" not in content and content.strip() == "Checking both."
    calls = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls", [])]
    assert [c["index"] for c in calls] == [0, 1]
    assert calls[1]["function"] == {"name": "read", "arguments": '{"filePath": "b.py"}'}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_tool_results_continue_the_same_thread(client, auth, fake) -> None:
    def agent(text: str):
        found = tool_result(text)
        return [Say(f"{found[1]} returned: {found[2]}")] if found else [Say(block(("read", {"filePath": "a.py"})))]

    fake.behaviour = agent
    first = await chat(client, auth, [{"role": "system", "content": "env: monday"}, user("read a.py")])
    thread_id = first.headers["x-hoplite-thread-id"]
    call = first.json()["choices"][0]["message"]["tool_calls"][0]

    # Agents regenerate their system prompt and re-serialize arguments: it is still the same conversation.
    echoed = {"role": "assistant", "content": "",
              "tool_calls": [{**call, "function": {"name": "read", "arguments": '{ "filePath" : "a.py" }'}}]}
    second = await chat(client, auth, [
        {"role": "system", "content": "env: tuesday"}, user("read a.py"), echoed,
        {"role": "tool", "tool_call_id": call["id"], "content": "print('hi')"},
    ])
    assert second.status_code == 200, second.text
    assert second.headers["x-hoplite-thread-id"] == thread_id
    assert second.json()["choices"][0]["finish_reason"] == "stop"
    assert second.json()["choices"][0]["message"]["content"] == "read returned: print('hi')"
    assert len(fake.thread_creations()) == 1
    assert appended_messages(fake)[-1].startswith(
        f'<tool_result id="{call["id"]}" name="read">\nprint(\'hi\')\n</tool_result>\n\n(Reminder:')


async def test_full_history_is_replayed_when_the_thread_is_unknown(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Say("done")]
    history = [
        user("rename x"),
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"filePath": "x.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "x = 1"},
    ]
    resp = await chat(client, auth, history)
    assert resp.json()["choices"][0]["message"]["content"] == "done"
    prompt = fake.thread_creations()[0]["prompt"]
    assert 'Assistant: <tool_calls>\n[{"name": "read", "arguments": {"filePath": "x.py"}}]\n</tool_calls>' in prompt
    assert 'Latest input:\n\n<tool_result id="call_1" name="read">\nx = 1\n</tool_result>' in prompt


async def test_invalid_calls_are_repaired(client, auth, fake) -> None:
    replies = iter([
        Say('<tool_calls>[{"name": "read", "arguments": {"filePath": "a.py"}</tool_calls>'),  # broken JSON
        Say(block(("delete_all", {}))),  # unknown tool
        Say(block(("read", {"filePath": "a.py"}))),
    ])
    fake.behaviour = lambda text: [next(replies)]
    resp = await chat(client, auth, [user("x")])
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
    repairs = appended_messages(fake)
    assert len(repairs) == 2
    assert "not valid JSON" in repairs[0] and "unknown tool 'delete_all'" in repairs[1]
    assert len(fake.thread_creations()) == 1


async def test_sandbox_tool_use_is_corrected(client, auth, fake) -> None:
    replies = iter([
        [Tool("read", "invalid_tool_input: Model tried to call unavailable tool read"), Say("Could not read it.")],
        [Tool("Glob", "No files found"), Say("The file does not exist.")],
        [Say(block(("read", {"filePath": "notes.txt"})))],
    ])
    fake.behaviour = lambda text: next(replies)
    resp = await chat(client, auth, [user("what is in notes.txt?")])
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
    repairs = appended_messages(fake)
    assert "tried to call the client tool `read` as a built-in function" in repairs[0]
    assert "you used your built-in tools (Glob)" in repairs[1]


async def test_gives_up_after_the_repair_budget(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Say("Trying.\n" + block(("edit", {"filePath": "a.py"})))]  # missing arguments
    resp = await chat(client, auth, [user("x")])
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"]
    assert "missing required arguments: oldString, newString" in choice["message"]["content"]
    assert len(appended_messages(fake)) == 2


async def test_tool_choice(client, auth, fake) -> None:
    fake.behaviour = lambda text: ([Say(block(("read", {"filePath": "a.py"})))] if "had to call" in text
                                   else [Say("No tool needed.")])
    required = await chat(client, auth, [user("x")], tool_choice="required")
    assert required.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert "For this reply you must call at least one client tool." in fake.thread_creations()[-1]["prompt"]

    fake.behaviour = lambda text: [Say(block(("read", {"filePath": "a.py"})))]
    named = await chat(client, auth, [user("y")], tool_choice={"type": "function", "function": {"name": "edit"}})
    assert named.json()["choices"][0]["finish_reason"] == "stop"  # never called `edit`, repairs exhausted
    assert "this reply must call `edit`" in named.json()["choices"][0]["message"]["content"]

    unknown = await chat(client, auth, [user("z")], tool_choice={"type": "function", "function": {"name": "nope"}})
    assert unknown.status_code == 400


async def test_plain_answers_and_parallel_limit(client, auth, fake) -> None:
    fake.behaviour = lambda text: [Say("Nothing to do.")]
    plain = await chat(client, auth, [user("hi")])
    assert plain.json()["choices"][0] == {"index": 0, "logprobs": None, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": "Nothing to do.",
                                                      "refusal": None}}

    fake.behaviour = lambda text: [Say(block(("read", {"filePath": "a"}), ("read", {"filePath": "b"})))]
    single = await chat(client, auth, [user("two files")], parallel_tool_calls=False)
    assert len(single.json()["choices"][0]["message"]["tool_calls"]) == 1
    assert "Put exactly one call in the array." in fake.thread_creations()[-1]["prompt"]

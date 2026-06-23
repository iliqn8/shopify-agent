"""
Dev Agent — reads/modifies agent files and pushes to GitHub → Railway auto-redeploys.
"""
import anthropic
import os
import base64
import json
import requests as _req

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = "iliqn8/shopify-agent"
GITHUB_BRANCH = "main"
GITHUB_API = "https://api.github.com"

EDITABLE_FILES = [
    "server.py",
    "claude_agent.py",
    "shopify_client.py",
    "knowledge_base.py",
    "dev_agent.py",
    "imagegen_server.py",
    "prompts.py",
    "local_control.py",
    "templates/app.html",
    "requirements.txt",
    "railway.toml",
]

TOOLS = [
    {
        "name": "list_files",
        "description": "List all editable agent files",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_file",
        "description": "Read the current content of an agent file from GitHub",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write new content to a file and push to GitHub (triggers redeploy)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "commit_message": {"type": "string"},
            },
            "required": ["path", "content", "commit_message"],
        },
    },
]


def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _read_github_file(path):
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = _req.get(url, headers=_gh_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def _write_github_file(path, content, commit_message, sha):
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}"
    body = {
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "sha": sha,
        "branch": GITHUB_BRANCH,
    }
    r = _req.put(url, headers=_gh_headers(), json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def run_tool(name, inputs):
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN not set in Railway Variables."}
    try:
        if name == "list_files":
            return {"files": EDITABLE_FILES}
        elif name == "read_file":
            path = inputs["path"]
            if path not in EDITABLE_FILES:
                return {"error": f"Not in editable list: {path}"}
            content, sha = _read_github_file(path)
            return {"path": path, "content": content, "sha": sha}
        elif name == "write_file":
            path = inputs["path"]
            if path not in EDITABLE_FILES:
                return {"error": f"Not in editable list: {path}"}
            _, sha = _read_github_file(path)
            result = _write_github_file(path, inputs["content"], inputs["commit_message"], sha)
            return {
                "success": True,
                "commit": result["commit"]["sha"][:7],
                "deployed": True,
            }
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}


def _serialize_content(content):
    """Convert Anthropic SDK objects to plain JSON-serializable dicts."""
    if not isinstance(content, list):
        return content
    result = []
    for block in content:
        if hasattr(block, "model_dump"):
            result.append(block.model_dump())
        elif isinstance(block, dict):
            result.append(block)
        else:
            result.append(str(block))
    return result


def _serialize_messages(messages):
    return [
        {"role": m["role"], "content": _serialize_content(m["content"])}
        for m in messages
    ]


def _to_sdk_messages(messages):
    """Convert serialized messages back to format Anthropic SDK accepts."""
    result = []
    for m in messages:
        content = m["content"]
        # If content is a list of dicts with 'type', Anthropic SDK accepts them as-is
        result.append({"role": m["role"], "content": content})
    return result


def chat(messages):
    system = """You are the Dev Agent — an expert AI developer embedded inside the Shopify AI Agent application.
You have full read/write access to all source files of this app via GitHub.

The app is a Flask web application hosted on Railway with these components:
- server.py — main Flask server, all API routes
- claude_agent.py — Claude AI integration with Shopify + image generation tools
- shopify_client.py — Shopify Admin API wrapper
- knowledge_base.py — SQLite training data and chat history
- dev_agent.py — this file (you)
- imagegen_server.py — AI product image generator (OpenAI gpt-image-2)
- templates/app.html — the entire frontend UI (chat, training, image gen, dev tabs)
- local_control.py — local PC computer control via ngrok

Your job: understand what the user wants, read the relevant files, make the changes, and push to GitHub. Railway auto-redeploys in ~1-2 minutes.

Rules:
- Always read the current file before editing it
- Make precise, minimal changes — don't rewrite unnecessarily
- You can read multiple files to understand context before making changes
- After a successful push, tell the user exactly what you changed and that Railway is redeploying
- If the user's request is unclear, ask ONE clarifying question
- All UI text must stay in English
- Be conversational and explain your reasoning"""

    sdk_messages = _to_sdk_messages(messages)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=system,
        tools=TOOLS,
        messages=sdk_messages,
    )

    while response.stop_reason == "tool_use":
        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type == "tool_use":
                result = run_tool(block.name, dict(block.input))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        messages = messages + [
            {"role": "assistant", "content": _serialize_content(assistant_content)},
            {"role": "user", "content": tool_results},
        ]

        sdk_messages = _to_sdk_messages(messages)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=system,
            tools=TOOLS,
            messages=sdk_messages,
        )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    if not text:
        text = "Done. Railway is redeploying with the changes (~1-2 min)."

    messages = messages + [
        {"role": "assistant", "content": _serialize_content(response.content)},
    ]

    return text, _serialize_messages(messages)

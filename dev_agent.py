"""
Dev Agent — reads/modifies agent files and pushes to GitHub → Railway auto-redeploys.
Requires GITHUB_TOKEN env var with repo write access.
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
                "path": {"type": "string", "description": "File path, e.g. server.py or templates/app.html"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write new content to an agent file and push to GitHub (triggers redeploy)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Full new file content"},
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
        return {"error": "GITHUB_TOKEN not set. Add it in Railway Variables."}

    try:
        if name == "list_files":
            return {"files": EDITABLE_FILES}

        elif name == "read_file":
            path = inputs["path"]
            if path not in EDITABLE_FILES:
                return {"error": f"File not in editable list: {path}"}
            content, sha = _read_github_file(path)
            return {"path": path, "content": content, "sha": sha}

        elif name == "write_file":
            path = inputs["path"]
            if path not in EDITABLE_FILES:
                return {"error": f"File not in editable list: {path}"}
            _, sha = _read_github_file(path)
            result = _write_github_file(
                path, inputs["content"], inputs["commit_message"], sha
            )
            return {
                "success": True,
                "commit": result["commit"]["sha"][:7],
                "message": "Pushed to GitHub. Railway is redeploying (1-2 min).",
            }

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}


def chat(messages):
    system = """You are a developer agent that modifies the Shopify AI Agent's own code.
You can read and write the agent's source files on GitHub. Changes trigger an automatic Railway redeploy.

Rules:
- Always read the current file FIRST before making changes
- Make minimal, targeted changes — don't rewrite whole files unnecessarily
- After writing, tell the user what changed and that Railway is redeploying (takes ~1-2 min)
- Only edit files in the editable list
- The app is in English — keep all UI text and code in English
- Respond in English"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8096,
        system=system,
        tools=TOOLS,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        assistant_content = response.content

        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, dict(block.input))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        messages = messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    if not text:
        text = "Done. Railway is redeploying with the changes (1-2 min)."

    return text, messages

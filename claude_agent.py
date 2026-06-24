import anthropic
import json
import os
import subprocess
import base64
import requests as _requests

import shopify_client as sc
import skills_loader

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

IMAGE_GEN_URL = os.getenv("IMAGE_GENERATOR_URL", "http://localhost:5001")


# ── Image Generator helpers ────────────────────────────────────────────────

def ensure_image_generator():
    return _ensure_image_generator_running()


def _ensure_image_generator_running():
    """Start image generator if not already running on port 5000."""
    try:
        _requests.get(IMAGE_GEN_URL, timeout=2)
        return True
    except Exception:
        pass
    img_gen_path = r"C:\Users\Iliyan\Desktop\image-generator\server.py"
    subprocess.Popen(
        ["python", img_gen_path],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        cwd=os.path.dirname(img_gen_path),
    )
    import time
    for _ in range(15):
        time.sleep(1)
        try:
            _requests.get(IMAGE_GEN_URL, timeout=1)
            return True
        except Exception:
            pass
    return False


def _build_brand_dna(product_title, domain_name="bambyna.com", competitor="",
                     color_preferences="", additional_notes="", image_b64=None, image_filename=None):
    data = {
        "product_title": product_title,
        "domain_name": domain_name,
        "competitor": competitor,
        "color_preferences": color_preferences,
        "additional_notes": additional_notes,
    }
    files = {}
    if image_b64 and image_filename:
        import io
        img_bytes = base64.b64decode(image_b64)
        files["product_image"] = (image_filename, io.BytesIO(img_bytes), "image/jpeg")
    resp = _requests.post(f"{IMAGE_GEN_URL}/build-brand-dna", data=data, files=files or None)
    resp.raise_for_status()
    return resp.json()


def _generate_images_sync(brand_dna, on_progress=None):
    """Consume SSE stream and return list of generated image URLs."""
    resp = _requests.post(
        f"{IMAGE_GEN_URL}/generate-images",
        json=brand_dna,
        stream=True,
        timeout=300,
    )
    urls = []
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if text.startswith("data:"):
            try:
                ev = json.loads(text[5:].strip())
                if ev.get("type") == "image_done":
                    urls.append({"index": ev["index"], "name": ev["name"],
                                 "url": IMAGE_GEN_URL + ev["url"]})
                    if on_progress:
                        on_progress(f"🖼️ Image {ev['index'] + 1} generated...")
                elif ev.get("type") == "progress" and on_progress:
                    on_progress(f"🎨 {ev.get('message', 'Generating...')}")
            except Exception:
                pass
    return urls


def _upload_image_to_shopify(product_id, image_url):
    resp = _requests.get(image_url, timeout=30)
    resp.raise_for_status()
    b64 = base64.b64encode(resp.content).decode()
    filename = image_url.split("/")[-1]
    r = _requests.post(
        f"https://{sc.SHOP}/admin/api/2024-04/products/{product_id}/images.json",
        headers=sc.HEADERS,
        json={"image": {"attachment": b64, "filename": filename}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["image"]


# ── Tools definition ───────────────────────────────────────────────────────

TOOLS = [
    # Shopify
    {
        "name": "get_shop_info",
        "description": "Get basic store information",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_products",
        "description": "List products from the store",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "title": {"type": "string"},
            },
        },
    },
    {
        "name": "get_product",
        "description": "Get details of a specific product by ID",
        "input_schema": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
    },
    {
        "name": "create_product",
        "description": "Create a new product in the store",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body_html": {"type": "string"},
                "vendor": {"type": "string"},
                "product_type": {"type": "string"},
                "tags": {"type": "string"},
                "price": {"type": "string"},
                "compare_at_price": {"type": "string"},
                "sku": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "update_product",
        "description": "Update an existing product",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "title": {"type": "string"},
                "body_html": {"type": "string"},
                "vendor": {"type": "string"},
                "tags": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "draft", "archived"]},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "delete_product",
        "description": "Delete a product by ID",
        "input_schema": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
    },
    {
        "name": "list_collections",
        "description": "List all collections",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_orders",
        "description": "List orders",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "status": {"type": "string"},
            },
        },
    },
    {
        "name": "list_pages",
        "description": "List all pages in the store",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_page",
        "description": "Create a new page in the store",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body_html": {"type": "string"},
            },
            "required": ["title"],
        },
    },
    # Image Generator
    {
        "name": "generate_product_images",
        "description": "Generate 9 professional product images using AI. Returns image URLs. Takes 2-3 minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_title": {"type": "string", "description": "Product name"},
                "domain_name": {"type": "string", "description": "Store domain, default bambyna.com"},
                "competitor": {"type": "string", "description": "Competitor site for style reference"},
                "color_preferences": {"type": "string", "description": "Preferred colors"},
                "additional_notes": {"type": "string", "description": "Extra instructions"},
                "image_b64": {"type": "string", "description": "Base64-encoded product photo uploaded by user (optional)"},
                "image_filename": {"type": "string", "description": "Filename of the uploaded photo (optional)"},
            },
            "required": ["product_title"],
        },
    },
    {
        "name": "upload_images_to_product",
        "description": "Upload generated images to a Shopify product",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image URLs to upload",
                },
            },
            "required": ["product_id", "image_urls"],
        },
    },
    # Theme
    {
        "name": "list_theme_files",
        "description": "List all files in the active Shopify theme (templates, sections, snippets, assets, config, layout)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_theme_file",
        "description": "Read the full content of a theme file by its key (e.g. 'sections/product-template.liquid', 'templates/product.json', 'assets/base.css')",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "File path in the theme, e.g. sections/main-product.liquid"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "update_theme_file",
        "description": "Write/update a theme file. Use get_theme_file first to read the current content, make the targeted changes, then write the full updated content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "File path in the theme"},
                "value": {"type": "string", "description": "Full new content of the file"},
            },
            "required": ["key", "value"],
        },
    },
    # Computer Control
    {
        "name": "run_command",
        "description": "Run a PowerShell command on the user's computer and return output",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "PowerShell command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "open_application",
        "description": "Open an application or file on the computer",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "App name or file path (e.g. notepad, chrome, C:\\file.txt)"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
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
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "get_running_processes",
        "description": "Get list of currently running applications/processes",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Human-readable tool names for status display
TOOL_LABELS = {
    "get_shop_info": "🏪 Getting store info...",
    "list_products": "📦 Loading products...",
    "get_product": "📦 Getting product details...",
    "create_product": "✏️ Creating product...",
    "update_product": "✏️ Updating product...",
    "delete_product": "🗑️ Deleting product...",
    "list_collections": "📁 Loading collections...",
    "list_orders": "📋 Loading orders...",
    "list_pages": "📄 Loading pages...",
    "create_page": "📄 Creating page...",
    "list_theme_files": "🎨 Loading theme files...",
    "get_theme_file": "📄 Reading theme file...",
    "update_theme_file": "💾 Updating theme file...",
    "generate_product_images": "🎨 Starting image generation...",
    "upload_images_to_product": "⬆️ Uploading images to Shopify...",
    "run_command": "💻 Running command...",
    "open_application": "🖥️ Opening application...",
    "list_files": "📂 Listing files...",
    "read_file": "📖 Reading file...",
    "write_file": "💾 Writing file...",
    "get_running_processes": "🖥️ Getting running processes...",
}


# ── Tool runner ────────────────────────────────────────────────────────────

def run_tool(name, inputs, on_progress=None):
    try:
        # Shopify tools
        if name == "get_shop_info":
            return sc.get_shop()
        elif name == "list_products":
            return sc.list_products(**inputs)
        elif name == "get_product":
            return sc.get_product(inputs["product_id"])
        elif name == "create_product":
            return sc.create_product(**inputs)
        elif name == "update_product":
            pid = inputs.pop("product_id")
            return sc.update_product(pid, **inputs)
        elif name == "delete_product":
            return sc.delete_product(inputs["product_id"])
        elif name == "list_collections":
            return sc.list_collections()
        elif name == "list_orders":
            return sc.list_orders(**inputs)
        elif name == "list_pages":
            return sc.list_pages()
        elif name == "create_page":
            return sc.create_page(**inputs)

        # Theme tools
        elif name == "list_theme_files":
            theme = sc.get_active_theme()
            if not theme:
                return {"error": "No active theme found"}
            files = sc.list_theme_files(theme["id"])
            return {"theme_name": theme["name"], "theme_id": theme["id"],
                    "files": [f["key"] for f in files]}
        elif name == "get_theme_file":
            theme = sc.get_active_theme()
            if not theme:
                return {"error": "No active theme found"}
            return sc.get_theme_file(theme["id"], inputs["key"])
        elif name == "update_theme_file":
            theme = sc.get_active_theme()
            if not theme:
                return {"error": "No active theme found"}
            return sc.update_theme_file(theme["id"], inputs["key"], inputs["value"])

        # Image Generator tools
        elif name == "generate_product_images":
            _ensure_image_generator_running()
            if on_progress:
                on_progress("🎨 Building brand DNA...")
            brand_dna = _build_brand_dna(
                product_title=inputs.get("product_title"),
                domain_name=inputs.get("domain_name", "bambyna.com"),
                competitor=inputs.get("competitor", ""),
                color_preferences=inputs.get("color_preferences", ""),
                additional_notes=inputs.get("additional_notes", ""),
                image_b64=inputs.get("image_b64"),
                image_filename=inputs.get("image_filename"),
            )
            if on_progress:
                on_progress("🖼️ Generating images (this takes 2-3 min)...")
            images = _generate_images_sync(brand_dna, on_progress=on_progress)
            return {"brand_dna": brand_dna, "images": images, "count": len(images)}

        elif name == "upload_images_to_product":
            results = []
            for i, url in enumerate(inputs.get("image_urls", [])):
                try:
                    if on_progress:
                        on_progress(f"⬆️ Uploading image {i+1}/{len(inputs['image_urls'])}...")
                    img = _upload_image_to_shopify(inputs["product_id"], url)
                    results.append({"id": img["id"], "src": img["src"]})
                except Exception as e:
                    results.append({"error": str(e), "url": url})
            return {"uploaded": results}

        # Computer Control tools — route to local agent if available
        elif name in ("run_command", "open_application", "list_files",
                      "read_file", "write_file", "get_running_processes"):
            local_url = os.getenv("LOCAL_AGENT_URL", "").rstrip("/")
            if local_url:
                try:
                    r = _requests.post(
                        f"{local_url}/run-tool",
                        json={"tool": name, "inputs": inputs},
                        timeout=30,
                    )
                    return r.json()
                except Exception as e:
                    return {"error": f"Local agent unreachable: {e}"}
            # Fallback: run locally (when not on Railway)
            if name == "run_command":
                result = subprocess.run(["powershell", "-Command", inputs["command"]],
                                        capture_output=True, text=True, timeout=30)
                return {"stdout": result.stdout[:3000], "stderr": result.stderr[:1000]}
            elif name == "open_application":
                subprocess.Popen(["start", inputs["target"]], shell=True)
                return {"opened": inputs["target"]}
            elif name == "list_files":
                result = subprocess.run(
                    ["powershell", "-Command", f"Get-ChildItem '{inputs['path']}' | Select-Object Name, Length, LastWriteTime | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=10)
                return {"files": result.stdout[:3000]}
            elif name == "read_file":
                with open(inputs["path"], "r", encoding="utf-8", errors="ignore") as f:
                    return {"content": f.read(5000)}
            elif name == "write_file":
                with open(inputs["path"], "w", encoding="utf-8") as f:
                    f.write(inputs["content"])
                return {"written": inputs["path"]}
            elif name == "get_running_processes":
                result = subprocess.run(
                    ["powershell", "-Command", "Get-Process | Sort-Object CPU -Descending | Select-Object -First 30 Name, CPU, WorkingSet | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=10)
                return {"processes": result.stdout[:3000]}

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}


# ── Main chat function (blocking, kept for compatibility) ──────────────────

def chat(messages, extra_context=""):
    system = _build_system(extra_context)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
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
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    return text, messages


# ── Streaming chat function (yields events) ───────────────────────────────

def chat_stream(messages, extra_context=""):
    """
    Generator that yields event dicts:
      {"type": "status", "text": "..."} — live progress update
      {"type": "done", "reply": "...", "messages": [...]} — final answer
    """
    # Extract user text from last message to detect relevant skills
    user_text = ""
    if messages:
        last = messages[-1]
        if isinstance(last.get("content"), str):
            user_text = last["content"]
        elif isinstance(last.get("content"), list):
            for block in last["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    user_text += block.get("text", "")

    skills_content = skills_loader.get_relevant_skills(user_text)
    system = _build_system(extra_context, skills_content)

    yield {"type": "status", "text": "🤔 Thinking..."}

    current_messages = list(messages)

    def _strip_images(msgs):
        """Replace base64 image blocks with placeholder to keep subsequent API calls lean."""
        result = []
        for msg in msgs:
            if isinstance(msg.get("content"), list):
                new_content = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "image":
                        new_content.append({"type": "text", "text": "[user attached image — already processed]"})
                    else:
                        new_content.append(block)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result

    def _blocks_to_dicts(content):
        """Convert Anthropic SDK content blocks to plain serializable dicts."""
        if not isinstance(content, list):
            return content
        result = []
        for block in content:
            if isinstance(block, dict):
                result.append(block)
            elif hasattr(block, "model_dump"):
                result.append(block.model_dump())
            elif hasattr(block, "dict"):
                result.append(block.dict())
            else:
                result.append({"type": getattr(block, "type", "unknown"),
                                "text": getattr(block, "text", str(block))})
        return result

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=current_messages,
            timeout=120.0,
        )

        iteration = 0
        while response.stop_reason == "tool_use" and iteration < 15:
            iteration += 1
            tool_results = []
            assistant_content = response.content

            for block in response.content:
                if block.type == "tool_use":
                    label = TOOL_LABELS.get(block.name, f"🔧 Running {block.name}...")
                    yield {"type": "status", "text": label}

                    progress_events = []

                    def on_progress(msg, _events=progress_events):
                        _events.append(msg)

                    result = run_tool(block.name, dict(block.input), on_progress=on_progress)

                    for msg in progress_events:
                        yield {"type": "status", "text": msg}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

            current_messages = _strip_images(current_messages) + [
                {"role": "assistant", "content": _blocks_to_dicts(assistant_content)},
                {"role": "user", "content": tool_results},
            ]

            yield {"type": "status", "text": "🤔 Processing results..."}

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=current_messages,
                timeout=120.0,
            )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        yield {"type": "done", "reply": text}

    except Exception as e:
        yield {"type": "done", "reply": f"Error: {str(e)}"}


def _build_system(extra_context="", skills_content=""):
    system = """You are an AI agent for managing a Shopify store and controlling the user's computer.
You have tools for: Shopify product/order/collection management, AI product image generation, and computer control (run commands, open apps, read/write files).
Always respond in English. Be concise and clear. When you complete a task, confirm what was done.

IMPORTANT — IMAGE GENERATION REQUESTS:
When the user asks to generate product images (or anything related to generating/creating images for a product), you MUST reply with EXACTLY this form — no more, no less. Do NOT call the tool yet. Do NOT rephrase the fields. Copy them exactly:

To generate product images, please provide the following details:

PRODUCT IMAGE * (upload using the 📎 button)
PRODUCT TITLE *
DOMAIN NAME *
COMPETITOR URL OR DESCRIPTION *
COLOR PREFERENCES (OPTIONAL)
ADDITIONAL NOTES (OPTIONAL)

Fields marked with * are required. You can upload a product photo using the 📎 button in the chat input.

Once the user replies with the details (and the product title is provided), call the generate_product_images tool immediately with all the information they gave you.
If the user already provided ALL required fields in their first message, call the tool immediately without showing the form."""

    if skills_content:
        system += f"\n\n## Active Marketing Skills (apply these for this task):\n{skills_content}"

    if extra_context:
        system += f"\n\n## Store Training Data:\n{extra_context}"

    return system

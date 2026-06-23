import anthropic
import json
import os
import subprocess
import base64
import requests as _requests

import shopify_client as sc

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


def _generate_images_sync(brand_dna):
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


# ── Tool runner ────────────────────────────────────────────────────────────

def run_tool(name, inputs):
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

        # Image Generator tools
        elif name == "generate_product_images":
            _ensure_image_generator_running()
            brand_dna = _build_brand_dna(
                product_title=inputs.get("product_title"),
                domain_name=inputs.get("domain_name", "bambyna.com"),
                competitor=inputs.get("competitor", ""),
                color_preferences=inputs.get("color_preferences", ""),
                additional_notes=inputs.get("additional_notes", ""),
                image_b64=inputs.get("image_b64"),
                image_filename=inputs.get("image_filename"),
            )
            images = _generate_images_sync(brand_dna)
            return {"brand_dna": brand_dna, "images": images, "count": len(images)}

        elif name == "upload_images_to_product":
            results = []
            for url in inputs.get("image_urls", []):
                try:
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


# ── Main chat function ─────────────────────────────────────────────────────

def chat(messages, extra_context=""):
    system = """You are an AI agent for managing a Shopify store and controlling the user's computer.
You have tools for: Shopify product/order/collection management, AI product image generation, and computer control (run commands, open apps, read/write files).
Always respond in English. Be concise and clear. When you complete a task, confirm what was done.

IMPORTANT — IMAGE GENERATION REQUESTS:
When the user asks to generate product images, ask them in plain conversational text for any missing details:
- Product name (required)
- Store domain (default: bambyna.com)
- Competitor site for style reference (optional)
- Color preferences (optional)
- Any extra notes (optional)
- They can also upload a product photo using the 📎 button in the chat input.

Once you have the product name (and any other details they provide), call the generate_product_images tool.
If the user already provided enough info in their first message, call the tool immediately without asking again.
If they uploaded a photo, it will be included in the message as image_b64 and image_filename fields."""

    if extra_context:
        system += f"\n\n## Store Training Data:\n{extra_context}"

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

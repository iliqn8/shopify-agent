import io
import json
import os
import time
import uuid
import base64

from PIL import Image
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from openai import OpenAI

from prompts import IMAGE_TYPES, build_brand_dna_prompt, get_image_prompt

load_dotenv()

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GENERATED_FOLDER = os.path.join(BASE_DIR, "generated_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "imagegen_app.html"))


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


@app.route("/build-brand-dna", methods=["POST"])
def build_brand_dna():
    product_title = request.form.get("product_title", "")
    domain_name = request.form.get("domain_name", "")
    competitor = request.form.get("competitor", "")
    color_preferences = request.form.get("color_preferences", "")
    additional_notes = request.form.get("additional_notes", "")

    image_base64 = None
    media_type = "image/jpeg"
    image_path = None

    if "product_image" in request.files:
        file = request.files["product_image"]
        if file and file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            image_path = f"/uploads/{filename}"
            media_type = "image/png" if ext == "png" else "image/jpeg"
            with open(filepath, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

    content = []
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
        })
    content.append({
        "type": "text",
        "text": build_brand_dna_prompt(
            product_title, domain_name, competitor, color_preferences, additional_notes
        ),
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=2500,
        )
        brand_dna = json.loads(response.choices[0].message.content)
        brand_dna["product_image_path"] = image_path
        brand_dna["product_title"] = product_title
        brand_dna["domain_name"] = domain_name
        return jsonify(brand_dna)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _call_image_api(prompt, brand_dna, extra_image_b64=None):
    """Use images.edit() with a reference photo when available, else pure generation.
    Priority: extra_image_b64 (from feedback upload) > original product image > generate()."""
    img_bytes = None

    if extra_image_b64:
        img_bytes = base64.b64decode(extra_image_b64)
    else:
        product_url = brand_dna.get("product_image_path", "")
        if product_url and "/uploads/" in product_url:
            fname = product_url.split("/uploads/")[-1]
            candidate = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.exists(candidate):
                with open(candidate, "rb") as f:
                    img_bytes = f.read()

    if img_bytes:
        # Convert to PNG so the mimetype is always known and accepted
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        png_buf = io.BytesIO()
        pil_img.save(png_buf, format="PNG")
        png_buf.seek(0)

        response = client.images.edit(
            model="gpt-image-2",
            image=("image.png", png_buf, "image/png"),
            prompt=prompt,
            size="1024x1024",
            quality="medium",
            n=1,
        )
    else:
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size="1024x1024",
            quality="medium",
            n=1,
        )

    return base64.b64decode(response.data[0].b64_json)


@app.route("/generate-images", methods=["POST"])
def generate_images():
    brand_dna = request.json

    def stream():
        yield _sse({"type": "status", "message": "Starting generation..."})

        for i, image_type in enumerate(IMAGE_TYPES):
            if i == 5:
                yield _sse({"type": "status", "message": "Rate limit window — waiting 13 seconds..."})
                time.sleep(13)

            yield _sse({"type": "generating", "index": i, "name": image_type})

            try:
                prompt = get_image_prompt(i, brand_dna)
                img_bytes = _call_image_api(prompt, brand_dna)
                filename = f"img_{i}_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(GENERATED_FOLDER, filename)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                yield _sse({
                    "type": "image_done",
                    "index": i,
                    "url": f"/generated/{filename}",
                    "name": image_type,
                })
            except Exception as e:
                yield _sse({"type": "error", "index": i, "name": image_type, "message": str(e)})

        yield _sse({"type": "all_done"})

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/regenerate-image", methods=["POST"])
def regenerate_image():
    data = request.json
    index = data.get("index")
    brand_dna = data.get("brand_dna")
    feedback = data.get("feedback", "").strip()
    extra_image_b64 = data.get("reference_image_b64")

    try:
        prompt = get_image_prompt(index, brand_dna)
        if feedback:
            prompt += f"\n\nSPECIFIC ADJUSTMENT REQUESTED: {feedback}"
        img_bytes = _call_image_api(prompt, brand_dna, extra_image_b64=extra_image_b64)
        filename = f"img_{index}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(GENERATED_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        return jsonify({"url": f"/generated/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _sse(data):
    return f"data: {json.dumps(data)}\n\n"


def run_imagegen(port=5000):
    app.run(debug=False, port=port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    print("\n  Product Image Generator")
    print("  Running at http://localhost:5000\n")
    run_imagegen()

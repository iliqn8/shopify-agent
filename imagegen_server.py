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


def _dictionary_accent(color):
    """True when `color` is the Accent of a row in the vibe palette dictionary."""
    table = build_brand_dna_prompt("", "", "")
    accents = {cells[4].strip().upper()
               for cells in (line.split("|") for line in table.splitlines())
               if len(cells) == 8 and cells[4].strip().startswith("#")}
    return (color or "").strip().upper() in accents


# What a designer means by each colour word, as hue ranges in degrees. The web's
# named colours are not always that: "purple" is #800080, which reads as magenta.
_HUES = [
    (("purple", "violet", "lilac", "lavender", "лилав"), "purple", 250, 292),
    (("magenta", "fuchsia", "pink", "розов"), "pink", 292, 345),
    (("red", "червен"), "red", 345, 15),
    (("orange", "оранжев"), "orange", 15, 40),
    (("yellow", "gold", "жълт"), "yellow", 40, 65),
    (("green", "зелен"), "green", 65, 160),
    (("teal", "cyan", "turquoise", "тюркоаз"), "teal", 160, 200),
    (("blue", "navy", "син"), "blue", 200, 250),
]
_FROM_IMAGE = ("image", "product", "photo", "picture", "снимк", "продукт")


def _hsv(color):
    import colorsys
    c = (color or "").strip().lstrip("#")
    if len(c) != 6:
        return None
    try:
        r, g, b = (int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


def _in_range(hue, lo, hi):
    return lo <= hue < hi if lo < hi else hue >= lo or hue < hi


def _family(color):
    hsv = _hsv(color)
    if not hsv or hsv[1] < 0.25 or hsv[2] < 0.15:
        return None
    return next((name for _, name, lo, hi in _HUES if _in_range(hsv[0], lo, hi)), None)


def _image_colors(path, limit=5):
    """The colours actually in the photo, one per hue family, most common first.

    Pixels are read one by one: averaging regions first turns a bright purple
    glow on a black box into a near-black purple no one would pick as an accent.
    """
    import colorsys
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((200, 200))
        groups = {}
        for r, g, b in img.getdata():
            h, sat, val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if sat < 0.35 or val < 0.3:
                continue
            name = next((n for _, n, lo, hi in _HUES if _in_range(h * 360, lo, hi)), None)
            if name:
                groups.setdefault(name, []).append((r, g, b))
        total = img.width * img.height
        found = []
        for name, pixels in groups.items():
            if len(pixels) < total * 0.002:
                continue
            # A real pixel from the brighter part of the group: the colour as
            # it shows, not as it fades into the shadows around it.
            pixels.sort(key=lambda p: max(p))
            rgb = pixels[len(pixels) * 3 // 4]
            found.append((len(pixels), "#%02X%02X%02X" % rgb, name,
                          max(1, round(100 * len(pixels) / total))))
        found.sort(reverse=True)
        return [(c, fam, share) for _, c, fam, share in found[:limit]]
    except Exception:
        return []


def _color_target(preference, image_colors):
    """The hue family asked for, and the photo's own shades of it if it has any."""
    text = (preference or "").lower()
    rule = next((r for r in _HUES if any(w in text for w in r[0])), None)
    from_image = any(w in text for w in _FROM_IMAGE)
    if not rule:
        # "as on the product" with no colour word: the photo's main colour.
        if from_image and image_colors:
            name = image_colors[0][1]
            rule = next(r for r in _HUES if r[1] == name)
        else:
            return None
    _, name, lo, hi = rule
    shades = [c for c, fam, _ in image_colors if fam == name]
    return {"name": name, "lo": lo, "hi": hi, "shades": shades}


def _color_facts(target, image_colors):
    lines = []
    if image_colors:
        lines.append("Colors measured in the attached product photo (hex, hue family, share of photo): "
                     + ", ".join("%s %s %d%%" % c for c in image_colors) + ".")
    if target:
        lines.append("The requested color is %s: hue %d-%d degrees." % (target["name"], target["lo"], target["hi"]))
        if target["shades"]:
            lines.append("The product's own %s is %s. Build ACCENT_COLOR and DARK_ACCENT_COLOR "
                         "from it (keep its hue, adjust lightness)." % (target["name"], ", ".join(target["shades"])))
        lines.append("Check the hue of every hex you return for it, not just its name: a standard "
                     "web color can sit outside that range (web \"purple\" #800080 is 300 degrees and looks pink).")
    return "\n".join(lines)


def _off_hue(brand_dna, target):
    """The accent fields whose hue is not the colour that was asked for."""
    if not target:
        return []
    return [key for key in ("ACCENT_COLOR", "DARK_ACCENT_COLOR")
            if _family(brand_dna.get(key)) != target["name"]]


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

    image_colors = _image_colors(filepath) if image_base64 else []
    target = _color_target(color_preferences, image_colors) if color_preferences else None

    content = []
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
        })
    content.append({
        "type": "text",
        "text": build_brand_dna_prompt(
            product_title, domain_name, competitor, color_preferences, additional_notes,
            color_facts=_color_facts(target, image_colors) if color_preferences else "",
        ),
    })

    try:
        messages = [{"role": "user", "content": content}]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=2500,
        )
        raw = response.choices[0].message.content
        brand_dna = json.loads(raw)

        # The prompt says the user's colors win, but the model has copied a
        # dictionary row, or picked a hex by its name whose hue is another color
        # ("purple" #800080 reads pink). Either way, ask once more with the reason.
        complaints = []
        if color_preferences and _dictionary_accent(brand_dna.get("ACCENT_COLOR")):
            complaints.append(f"ACCENT_COLOR {brand_dna.get('ACCENT_COLOR')} is copied from the "
                              "palette dictionary.")
        for key in _off_hue(brand_dna, target):
            hsv = _hsv(brand_dna.get(key))
            seen = "%d degrees" % hsv[0] if hsv else "not a valid hex"
            complaints.append(f"{key} {brand_dna.get(key)} is {seen}, which reads as "
                              f"{_family(brand_dna.get(key)) or 'grey'}, not {target['name']} "
                              f"({target['lo']}-{target['hi']} degrees).")
        if complaints:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    " ".join(complaints) + f" The user asked for: {color_preferences}. "
                    + (f"The product's own {target['name']}: {', '.join(target['shades'])}. "
                       if target and target["shades"] else "")
                    + "Return the same JSON with all five colors rebuilt from that preference."},
            ]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
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

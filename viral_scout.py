"""Claude scores CJ products for viral potential — the "trained" half of Product Hunter.

CJ knows prices, listings and dates. It knows nothing about whether a product
stops a thumb mid-scroll. That judgement comes from Claude looking at the photo
and the name, scored trait by trait, for only the traits the user switched on.

"Training" is two things, both plain data rather than fine-tuning:
- the traits and the free-text criteria the user picks for this run, and
- products they marked 👍 / 👎 earlier, with their notes, sent along as
  examples of their taste. The more they mark, the closer the scores get.

The overall score is computed HERE from the trait scores, not asked of the
model, so it always means "average of what you asked for" and nothing else.
"""

import io
import os
import json
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import requests
from PIL import Image

MODEL = "claude-opus-5"
claude = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# Opus 5 list price per token.
PRICE_IN = 5.00 / 1_000_000
PRICE_OUT = 25.00 / 1_000_000

BATCH = 10          # products per Claude call
PARALLEL = 3        # Claude calls at once
MAX_PRODUCTS = 60
THUMB = 384         # px — ~200 image tokens; enough to judge a product, cheap to send

# Order matters: it is the order of the chips in the UI.
TRAITS = {
    "wow": ("Wow factor",
            "Makes someone stop scrolling within 2 seconds — surprising, clever, 'wait, what is that?'"),
    "controversy": ("Controversy",
                    "Splits opinion or sparks comments/debate (weird, divisive, 'who would buy this?', "
                    "'this should be illegal') — without being unsafe or banned from ads"),
    "demo": ("Easy to demo on video",
             "The value is obvious in a short silent clip; satisfying or visual transformation"),
    "problem": ("Solves a real problem",
                "Removes a pain people feel often and recognise instantly"),
    "before_after": ("Before / after",
                     "Has a clear visual before-and-after or transformation moment"),
    "impulse": ("Impulse price",
                "Feels like a no-brainer at a $20–40 retail price; perceived value well above cost"),
    "novelty": ("Not in retail stores",
                "People have not seen it in supermarkets, Amazon basics or local shops"),
    "broad": ("Broad audience",
              "Relevant to a large mass-market audience, not a tiny niche"),
    "gift": ("Gift potential",
             "Easy to buy for someone else; fun to give"),
    "emotional": ("Emotional trigger",
                  "Taps pets, kids, safety, beauty, nostalgia or cuteness"),
}
DEFAULT_TRAITS = ["wow", "controversy", "demo", "problem", "impulse"]

SYSTEM = """You are a senior e-commerce product researcher who finds winning products for TikTok and Meta dropshipping ads.

You will be shown CJ Dropshipping products (photo + facts) and score each one 0–10 on the traits the user chose. Be a harsh, calibrated judge: most products are ordinary and should score 2–5. Reserve 8–10 for products that would genuinely go viral. A score is about the product as an ad subject, not about photo quality — CJ photos are often poor.

"listings" is how many dropshippers have already added the product to their stores. A high number means proven demand but more competition; a low number means untested or early. Mention this in `why` when it matters.

If the user gave examples of products they liked or disliked, learn their taste from them and let it move your scores.

For each product write:
- `hook`: one concrete opening line or first shot for a short video ad (English).
- `why`: 1–2 sentences on what drives the score.
- `risks`: short — ad-policy problems, fragile/quality risk, trademark/IP, saturation, seasonality. "None obvious" if none.

Return every product you were given, with its exact id."""


def options():
    return {"traits": [{"key": k, "label": v[0], "hint": v[1]} for k, v in TRAITS.items()],
            "default_traits": DEFAULT_TRAITS,
            "max_products": MAX_PRODUCTS,
            # Measured: 10 products with photos, 5 traits, effort medium = $0.0744.
            "usd_per_product": 0.008}


def _thumb_b64(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img = img.convert("RGB")
        img.thumbnail((THUMB, THUMB))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        return base64.standard_b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _schema(traits):
    return {
        "type": "object",
        "properties": {"products": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "scores": {"type": "object",
                           "properties": {t: {"type": "integer"} for t in traits},
                           "required": list(traits), "additionalProperties": False},
                "hook": {"type": "string"},
                "why": {"type": "string"},
                "risks": {"type": "string"},
            },
            "required": ["id", "scores", "hook", "why", "risks"],
            "additionalProperties": False,
        }}},
        "required": ["products"],
        "additionalProperties": False,
    }


def _examples_text(feedback):
    if not feedback:
        return ""
    lines = []
    for label, verdict in (("LIKED — more like these", "like"), ("DISLIKED — avoid these", "dislike")):
        rows = [f for f in feedback if f["verdict"] == verdict][:20]
        if rows:
            lines.append(label + ":")
            for f in rows:
                note = f" — their note: {f['note']}" if f.get("note") else ""
                lines.append(f"- {f['name']} ({f.get('category') or 'no category'}, ${f.get('price') or '?'}){note}")
    return "\n".join(lines)


def _brief(traits, custom, feedback):
    parts = ["Score these traits (0–10 each):"]
    parts += [f"- {t}: {TRAITS[t][0]} — {TRAITS[t][1]}" for t in traits]
    if custom.strip():
        parts.append("\nThe user also wants you to look for / weigh this:\n" + custom.strip())
    ex = _examples_text(feedback)
    if ex:
        parts.append("\nThe user's taste, from products they judged before:\n" + ex)
    return "\n".join(parts)


def _call(content, schema):
    """One structured call. Returns (data, usage) or raises RuntimeError with a readable message."""
    last = None
    for attempt in range(1, 4):
        try:
            with claude.messages.stream(
                model=MODEL,
                max_tokens=32000,
                system=SYSTEM,
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}],
            ) as stream:
                msg = stream.get_final_message()
            if msg.stop_reason == "refusal":
                raise RuntimeError("Claude declined to score this batch.")
            if msg.stop_reason == "max_tokens":
                raise RuntimeError("Claude ran out of room on this batch.")
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return json.loads(text), msg.usage
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            last = e
            time.sleep(4 * attempt)
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API error {e.status_code}: {str(e)[:200]}")
        except json.JSONDecodeError:
            raise RuntimeError("Claude returned something that was not JSON.")
    raise RuntimeError(f"Claude is busy — try again in a minute ({type(last).__name__}).")


def _score_batch(batch, thumbs, brief, traits):
    content = [{"type": "text", "text": brief + "\n\nPRODUCTS:"}]
    for p in batch:
        facts = {"id": p["id"], "name": p.get("name"), "category": p.get("category"),
                 "cost_usd": p.get("price"), "listings": p.get("listed")}
        content.append({"type": "text", "text": json.dumps(facts, ensure_ascii=False)})
        if thumbs.get(p["id"]):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": thumbs[p["id"]]}})
        else:
            content.append({"type": "text", "text": "(photo unavailable — judge from the name)"})
    data, usage = _call(content, _schema(traits))
    return data.get("products") or [], usage


def overall(scores, traits):
    vals = [max(0, min(10, int(scores.get(t, 0)))) for t in traits]
    return round(sum(vals) / len(vals) * 10) if vals else 0


def score_stream(products, traits, custom="", feedback=None):
    traits = [t for t in traits if t in TRAITS] or list(DEFAULT_TRAITS)
    products = [p for p in products if p.get("id")][:MAX_PRODUCTS]
    if not products:
        yield {"type": "done", "error": "No products to score."}
        return

    yield {"type": "status", "text": f"📷 Loading {len(products)} product photos…"}
    thumbs = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_thumb_b64, p.get("image")): p["id"] for p in products if p.get("image")}
        for f in as_completed(futs):
            thumbs[futs[f]] = f.result()

    brief = _brief(traits, custom or "", feedback or [])
    batches = [products[i:i + BATCH] for i in range(0, len(products), BATCH)]
    results, errors, usd, done = {}, [], 0.0, 0
    yield {"type": "status", "text": f"🔥 Claude is judging {len(products)} products…"}

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futs = [pool.submit(_score_batch, b, thumbs, brief, traits) for b in batches]
        for f in as_completed(futs):
            done += 1
            try:
                rows, usage = f.result()
                usd += usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT
                for r in rows:
                    scores = {t: max(0, min(10, int(r["scores"].get(t, 0)))) for t in traits}
                    results[str(r["id"])] = {"scores": scores, "overall": overall(scores, traits),
                                             "hook": r.get("hook", ""), "why": r.get("why", ""),
                                             "risks": r.get("risks", "")}
            except Exception as e:
                errors.append(str(e))
            yield {"type": "status",
                   "text": f"🔥 {done}/{len(batches)} batches scored · ${usd:.3f} so far"}

    if not results:
        yield {"type": "done", "error": errors[0] if errors else "Nothing was scored."}
        return
    missing = len(products) - len(results)
    yield {"type": "done", "results": results, "traits": traits, "usd": round(usd, 4),
           "warning": (f"{missing} product(s) were not scored: {errors[0]}" if missing and errors
                       else f"{missing} product(s) were not scored." if missing else "")}

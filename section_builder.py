import os
import re
import json
import base64
import tempfile
import requests
import cv2
from bs4 import BeautifulSoup
import anthropic
import knowledge_base as kb

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

SECTION_PROMPT_TEMPLATE = """ROLE
You are an expert Shopify Theme developer specializing in Online Store 2.0 (OS 2.0) sections, in the
same style and conventions as Shopify's own Dawn theme.

TASK
The user is showing you ONE section of a webpage from this reference URL:
[REFERENCE_URL]

You are given a DESKTOP screenshot of that section, and — if provided — a second MOBILE screenshot
showing exactly how the SAME section renders on an actual mobile device on the source site. Each
attached image is preceded by a text label identifying which one it is.

Below is the raw extracted text content of that page, for exact copy reference:
---PAGE TEXT START---
[PAGE_TEXT]
---PAGE TEXT END---

The page text above may include content from OTHER sections of the page too — ignore anything not
visibly present in the screenshot. Use the page text ONLY to source exact wording (headlines, body
copy, button labels) for the specific section shown in the screenshot; never invent or paraphrase
copy that contradicts the fetched text. If the page text looks empty, garbled, or unrelated to the
screenshot (e.g. a bot-block/JS-challenge page), fall back to reading text directly off the screenshot.

MERCHANT'S WRITTEN NOTES
The merchant may have added written clarifications about what to build or change — these are their
own words and take priority over your own visual interpretation whenever they conflict:
---NOTES START---
[MERCHANT_NOTES]
---NOTES END---

ANIMATION REFERENCE FRAMES
If frame images labeled "ANIMATION FRAME" are attached, they were extracted in chronological order
from a short screen recording of an animated/moving part of the reference section (e.g. a hover
effect, an auto-rotating carousel, a fade/slide transition, a marquee). Study how the element's
position/opacity/size changes ACROSS the frames in sequence, then recreate that motion with native
CSS (`@keyframes` + `animation`, or `transition` for hover/interaction states) scoped under the
section's own class — never with JavaScript. Wrap continuous/looping animations in
`@media (prefers-reduced-motion: reduce) { animation: none; }` so motion-sensitive visitors can
disable it. If no animation frames are attached, do not invent any motion/animation that isn't
visible in the static screenshots.

REAL EXTRACTED STYLES (STRICT, only when provided — ground truth, not a guess)
[PAGE_CONTEXT]
The values above (if present) were read directly from the live page's computed CSS via a real
browser — not estimated from a screenshot. Where they overlap with what you'd otherwise visually
estimate (font-family, font-size, color, background-color), USE THESE EXACT VALUES instead of
approximating from pixels. This is the single most reliable signal you have for typography and
color fidelity — prioritize it over your own visual judgement whenever both are available. If a
"font_face_names"/"font_links" list is present, that is the real font the site loads — reference
the same font name in your `font-family` declarations (falling back to
`var(--font-heading-family)`/`var(--font-body-family)` is still required per the rule below, but
knowing the real name helps you judge size/weight/spacing accurately even when the merchant's own
theme font differs).

WHAT TO BUILD
Analyze ONLY the section visible in the screenshot (ignore header/footer/other sections unless they
are literally inside the screenshot). Recreate it as a brand-new, custom Shopify section.

VIDEO CONTENT (STRICT)
Look carefully for signs that a card/tile is a VIDEO, not a static image: a small mute/unmute
speaker icon, a play-button overlay, a UGC/testimonial-style vertical (9:16) aspect ratio, or a
timeline/progress bar. If you see these, that block's media setting MUST be
`"type": "video"` (Shopify's native video picker, NOT `image_picker`). Render it with
`{{ block.settings.video | video_tag: muted: true, autoplay: false, loop: true, controls: false,
class: 'your-class' }}` — the `video_tag` filter ONLY, always. NEVER manually construct a `<video>`
tag with a hand-picked `src="{{ block.settings.video.sources[N].url }}"` — indexing into `.sources`
yourself is fragile (wrong index, wrong format, wrong MIME type) and frequently renders a broken,
invisible video even when a file is correctly assigned. `video_tag` handles sources/poster/mime
types correctly and is the only reliable way to render a Shopify-hosted video. Add a small
absolutely-positioned mute/unmute toggle button matching the reference — this requires a single
minimal inline `onclick` that toggles the adjacent `<video>` element's `.muted` property (this is a
basic UI control, not an "animation", so it's fine to use a tiny inline handler here even though
animations themselves must stay pure CSS). Give the block an `image_picker` fallback ONLY if the
reference clearly shows a static photo card (no play/mute icon) — never guess; when unsure between
image and video, prefer `video` if ANY play/mute affordance is visible.

FEATURED / CENTER ITEM IN A CAROUSEL (STRICT)
If the reference shows one card in a row visually emphasized (larger, centered, elevated) among
otherwise-equal siblings, do NOT expose a separate numeric "which position is featured" setting
(e.g. a `range` setting like `featured_index`) — merchants can't intuit what number to type, and it
silently goes stale the moment blocks are added, removed, or reordered. Instead, compute the true
middle position from the actual block count in Liquid and key the featured styling off that, e.g.:
```
{%- liquid
  assign block_count = section.blocks.size
  assign featured_position = block_count | plus: 1 | divided_by: 2
-%}
```
then `{% if forloop.index == featured_position %}` for the featured class. This way, whichever
block a merchant drags to the middle of the list (using Shopify's native block reordering, already
enabled via `block.shopify_attributes`) automatically becomes the visually featured one — no
separate setting to keep in sync, and reordering blocks is the ONLY control needed.

HORIZONTAL SCROLL CAROUSELS (STRICT)
If the screenshot shows a row of cards that is wider than the section (cards visibly cut off at
the right edge, or a scrollbar is visible), this is a horizontally-scrollable shelf — NOT a static
flex row that would silently clip the extra cards with no way to reach them. Build it as
`display: flex; overflow-x: auto; scroll-snap-type: x mandatory; gap: ...;` on the row container,
with each card as `flex: 0 0 auto; scroll-snap-align: start;` and an explicit card width so the
partial-next-card peek matches the screenshot. Include ALL items visible across the full width of
the reference (don't stop at only as many as fit in one viewport) as separate blocks.

NO INVENTED DECORATION (STRICT)
Only build elements that are actually visible in the screenshot. Do NOT add extra decorative icons,
floating bubbles, badges, or graphics that are not present in the reference image, even if you think
they would look nice. Every icon/graphic in your output must map to exactly ONE editable setting
(a block's image_picker or a section image_picker) — never render the same visual concept twice
(e.g. an icon inside a feature block AND a second copy of a similar icon floating elsewhere as a
hardcoded, non-editable SVG). If a piece of content doesn't map to something editable, leave it out
rather than hardcoding it.

EMPTY-STATE PLACEHOLDERS (STRICT)
The merchant will NOT have picked any images yet the first time this section is added (every
image_picker starts blank). Every image_picker MUST render a safe, simple fallback when blank —
a plain empty `<div>` styled with CSS only (e.g. `background: rgba(var(--color-foreground), 0.06);`
plus the element's normal width/height/border-radius) is enough. NEVER use the `placeholder_svg_tag`
filter chained with `append` filters (e.g. `{{ 'x' | placeholder_svg_tag: ... | append: ... }}`) —
`append` on the output of `placeholder_svg_tag` concatenates raw text (like the section ID) onto the
rendered SVG and prints it as visible garbled text on the page. If you use `placeholder_svg_tag` at
all, call it with exactly one plain string argument and nothing appended afterward. When in doubt,
skip `placeholder_svg_tag` entirely and use the empty-styled-`<div>` fallback instead — it always
renders cleanly with zero risk of leaking text. Apply the same rule consistently to EVERY
image_picker in the section (all block icons AND the main section image) — never give one a clean
fallback and another no fallback at all.

TYPOGRAPHY & SIZE FIDELITY (STRICT)
The goal is a PIXEL-FAITHFUL match to the screenshot, not a generic Dawn-style approximation:
- NEVER let text fall back to the browser's default font. Explicitly set
  `font-family: var(--font-heading-family);` on every heading-like element and
  `font-family: var(--font-body-family);` on every body/paragraph element, so the section always
  renders in the merchant's actual theme fonts instead of a mismatched generic font. This is
  mandatory even though it may seem redundant with inheritance — do it explicitly every time.
- Measure font sizes, spacing, and element proportions AGAINST the screenshot, not against a
  memorized "typical" scale. If the heading fills most of the section's width and wraps to 2 lines
  in the screenshot, its font-size must be large enough to do the same (do not default to a
  conservative/small clamp() range just because that's common in other Shopify sections). If body
  text reads as a normal comfortable paragraph size in the screenshot, don't shrink it down.
- Do NOT change text-align, flex-direction, or other structural properties at a breakpoint unless
  the corresponding screenshot (mobile screenshot for the 749px query, desktop screenshot otherwise)
  actually shows that change. A centered desktop heading stays centered on mobile unless the mobile
  screenshot clearly shows it left-aligned — never flip alignment as an unexplained "mobile default."

OUTPUT FORMAT (STRICT)
Output ONE thing only: a single, complete Shopify `.liquid` section file, wrapped in one ```liquid
code fence and nothing else — no explanation before or after the fence.

The file MUST contain, in this order:
1. HTML markup using semantic tags. Output EVERY piece of editable content via
   {{ section.settings.xxx }} / {{ block.settings.xxx }} — never hardcode text/images that a
   merchant would reasonably want to edit.
2. A <style> block. Give the outer wrapper element the class custom-section-{{ section.id }} and
   scope EVERY CSS rule under that exact class (never write bare/global selectors like `h2 {}` or
   `.button {}` — they will collide with the rest of the theme). section.id is unique per rendered
   instance, so this guarantees no collision even if this section is added twice on one page.
   COLOR RULE (STRICT): the outer wrapper must carry a `color-{{ section.settings.color_scheme }}`
   class (Shopify's scheme system), and every background/text/border color in your CSS that should
   follow the merchant's chosen Color Scheme MUST be written as `rgb(var(--color-background))`,
   `rgb(var(--color-foreground))`, or `rgba(var(--color-foreground), 0.NN)` for muted variants —
   NEVER a literal hex/rgb value like `#f0faff` or `#333` for these. Hardcoded literal colors make
   the "Color Scheme" setting do nothing when the merchant changes it, which is a critical bug.

   ACCENT COLORS MUST BE EDITABLE TOO (STRICT): color_scheme only covers the section's overall
   background/foreground — it does NOT give the merchant control over specific accent colors like
   a button's background, a highlighted headline word, star-rating color, badge/icon color, or a
   stat bubble's fill. For EVERY such distinct accent color you use, add its own
   `"type": "color"` setting to the schema (e.g. `button_background_color`, `button_text_color`,
   `heading_highlight_color`, `stat_bubble_color`, `stat_number_color`, `star_color` — name them for
   what they control), with `"default"` set to the exact hex value you observed in the screenshot.
   Apply each one by writing the Liquid variable directly into the CSS rule's value inside the
   `<style>` block, e.g. `background-color: {{ section.settings.button_background_color }};` —
   this renders server-side, so it works exactly like a hardcoded color visually while remaining
   fully editable by the merchant afterward. NEVER leave an accent color as a bare, unexposed
   literal hex in the CSS — if it's a real color decision (not pure structural CSS like border-radius
   or spacing), it needs a setting. This applies to every accent color, on every element, without
   exception — a merchant should be able to restyle any color in the section from the theme editor
   alone, with zero code edits.
3. Responsive breakpoints matching Dawn/OS 2.0 convention:
   - Unprefixed rules = desktop.
   - @media screen and (max-width: 989px) {{ ... }} = tablet adjustments.
   - @media screen and (max-width: 749px) {{ ... }} = mobile adjustments (stack multi-column
     layouts to one column here).
   MOBILE GROUND TRUTH (STRICT, only when a MOBILE screenshot was provided): the 749px media query
   MUST reproduce the exact element order, stacking, and grouping visible in the MOBILE screenshot —
   this is ground truth, not a guess. Do NOT default to a generic "image always first" or
   "image always last" stacking assumption. If the mobile screenshot shows content interleaved
   (e.g. some items before the image, some after), reproduce that exact order using CSS `order`
   on flex/grid children (keep the HTML/blocks in their natural editable sequence, and only reorder
   visually via CSS) rather than by physically reordering the HTML. If NO mobile screenshot was
   provided, fall back to sensible Dawn-style single-column stacking in the same top-to-bottom
   order the elements appear in the desktop screenshot.
   ROW-FITTING ARITHMETIC (STRICT): if the screenshot shows N repeated items (icons, stat bubbles,
   badges, columns, etc.) sitting side-by-side on ONE row, your mobile/tablet CSS sizes for that
   item MUST keep all N on one row at that breakpoint — do the arithmetic explicitly:
   (item width × N) + (gap × (N-1)) + container horizontal padding must be ≤ the breakpoint's
   viewport width (749px for mobile, 989px for tablet). If your first size choice doesn't fit,
   shrink the item width/gap until it does — never default to a size that "looks nice" in
   isolation and then let it silently wrap to fewer items per row than the screenshot shows.
   LINE-BREAK MATCHING (STRICT): for any heading/headline that wraps across multiple lines in a
   screenshot, choose a font-size (at that breakpoint) small enough that the SAME words fall on
   the same line as shown — do not default to a large generic hero font-size that forces an extra
   line break the reference doesn't have. When in doubt, size text conservatively (smaller) rather
   than aggressively (larger), since oversized text breaking the reference's line groups is a
   frequent, noticeable mismatch.
4. A {% schema %} block containing STRICTLY VALID JSON (double-quoted keys/strings, no trailing
   commas, no comments) with:
   - "name": short human-readable name shown in the Add Section menu.
   - "settings": a real editable setting for every non-repeating piece of content — use
     "image_picker" for images, "text" for short strings, "richtext"/"textarea" for longer
     copy, "color_scheme" for the section's colors (use section.settings.color_scheme with
     a color-{{ section.settings.color_scheme }} class — do NOT hardcode colors that should
     follow the merchant's theme), and "url" for any button/link targets.
   - "blocks": if there are ANY repeated items (icon rows, feature cards, testimonials, logos,
     steps), define block "type" entries each with their own "name" and "settings". In the
     Liquid body loop with {% for block in section.blocks %} and put {{ block.shopify_attributes }}
     on each block's outer element — this is REQUIRED for merchants to drag-reorder blocks in the
     theme editor.
   - "max_blocks": a sensible cap (e.g. 12) if blocks are used.
   - "presets": MUST include at least one preset object with a "name". A section with no presets
     will NOT appear in the theme editor's "Add section" picker at all.
   - Do NOT add a "templates" or "enabled_on" restriction — it should be addable on any template,
     exactly like native sections such as "Image with text".

SELF-CHECK BEFORE YOU SEND
- Output is ONLY the fenced ```liquid block, nothing else.
- Every visible text/image in the screenshot maps to an editable setting/block setting.
- Repeated items use blocks, each with {{ block.shopify_attributes }} on its outer element.
- No hardcoded decorative icons/graphics exist beyond what's visible in the screenshot, and no
  visual concept (e.g. an icon) is duplicated in two different hardcoded places.
- Every scheme-following color in the CSS uses rgb(var(--color-background)) /
  rgb(var(--color-foreground)) / rgba(var(--color-foreground), 0.NN) — zero hardcoded hex colors
  for backgrounds or text.
- Every distinct accent color (button, highlight, icon, bubble, star rating, etc.) has its own
  `"type": "color"` schema setting with a default matching the screenshot, applied via
  `{{ section.settings.xxx }}` directly in the CSS — zero bare unexposed hex accent colors anywhere.
- {% schema %} is valid JSON with a non-empty presets array.
- All CSS is scoped under .custom-section-{{ section.id }} — zero bare/global selectors.
- Both max-width: 989px and max-width: 749px media queries are present and meaningfully change
  the layout.
- If a MOBILE screenshot was provided, the 749px layout's element order/grouping matches it exactly
  (checked via CSS `order`, not by guessing).
- Every image_picker (block icons AND section image) has a clean CSS-only empty-state fallback —
  zero use of `placeholder_svg_tag` combined with `append` filters anywhere in the output.
- Every heading uses `font-family: var(--font-heading-family)` and every body/paragraph element uses
  `font-family: var(--font-body-family)` — never left to fall back on browser/generic defaults.
- Font sizes and proportions were measured against the screenshot, not defaulted to a generic scale.
- No text-align/flex-direction/structural change at any breakpoint unless the matching screenshot
  (mobile screenshot for 749px, desktop screenshot otherwise) actually shows that change.
- For every group of N repeated items shown on one row in a screenshot, did the math: item width
  × N + gaps + padding fits inside that breakpoint's viewport width — they won't silently wrap.
- Wrapped headlines break on the same words as the screenshot at each breakpoint — sizes were
  chosen conservatively, not defaulted to a large generic hero size.
- Any flex container centering a single number/icon/short text uses `align-items: center` (never
  `baseline`, which visibly pushes the content toward the top of a circle/bubble).
- Any card showing a mute/play icon or a vertical UGC-style video uses a block `"type": "video"`
  setting (never `image_picker`), rendered ONLY via the `video_tag` filter — zero manually
  constructed `src="{{ ...sources[N].url }}"` attributes anywhere.
- Any row of cards wider than the section is a real horizontally-scrollable shelf
  (`overflow-x: auto` + `scroll-snap`), including EVERY card visible across the screenshot's full
  width — not a static row that clips extras with no way to reach them.
- Any single visually-emphasized "featured" item among carousel siblings is keyed off a
  Liquid-computed true middle position (`block_count | plus: 1 | divided_by: 2`), never a separate
  manually-set numeric setting the merchant would have to keep in sync themselves.
"""


def fetch_page_text(url, max_chars=15000):
    """Fetch a URL and return (title, cleaned_visible_text). Never raises."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ShopifyAgentSectionBuilder/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        return None, f"[Could not fetch URL: {e}]"

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)[:max_chars]


def extract_video_frames(video_bytes, max_frames=6):
    """Extract up to max_frames evenly-spaced JPEG frames from raw video bytes.
    Returns a list of {b64, media_type} dicts, oldest-first. Never raises — returns [] on failure."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        count = min(max_frames, total)
        indices = [int(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                frames.append({"b64": base64.b64encode(buf.tobytes()).decode(), "media_type": "image/jpeg"})
        cap.release()
        return frames
    except Exception:
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _extract_liquid(text):
    m = re.search(r"```(?:liquid)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _enforce_heading_alignment(liquid_code):
    """Deterministic safety net: strip any text-align override on heading-like selectors
    inside @media blocks. Despite explicit prompt instructions, the model repeatedly flips
    headings to text-align:left on mobile with no basis in the reference screenshot — so this
    is enforced in code rather than relying on the prompt alone. Liquid {{ }} / {% %} tags are
    masked first so their braces don't confuse the CSS block matching below."""
    placeholders = []

    def mask(m):
        placeholders.append(m.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    masked = re.sub(r"\{\{.*?\}\}|\{%.*?%\}", mask, liquid_code, flags=re.DOTALL)

    def strip_rule(rule_match):
        selector, body = rule_match.group(1), rule_match.group(2)
        if "heading" in selector.lower():
            body = re.sub(r"\s*text-align\s*:\s*[^;]+;", "", body)
        return selector + "{" + body + "}"

    def strip_media_block(media_match):
        return re.sub(r"([^{}]+)\{([^{}]*)\}", strip_rule, media_match.group(0))

    fixed_masked = re.sub(r"@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", strip_media_block, masked, flags=re.DOTALL)

    def unmask(m):
        return placeholders[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", unmask, fixed_masked)


def _extract_schema_name(liquid_code):
    m = re.search(r"\{%\s*schema\s*%\}(.*?)\{%\s*endschema\s*%\}", liquid_code, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("name")
    except Exception:
        return None


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", (name or "custom-section").lower()).strip("-")[:40]
    return s or "custom-section"


def unique_asset_key(section_name):
    """sections/custom-{slug}.liquid, incrementing on collision against existing history."""
    base = slugify(section_name)
    existing = {row["asset_key"] for row in kb.list_custom_sections()}
    key = f"sections/custom-{base}.liquid"
    n = 2
    while key in existing:
        key = f"sections/custom-{base}-{n}.liquid"
        n += 1
    return key


def _format_page_context(page_context):
    """Turn a browser_capture.capture_page()-style computed_styles dict into readable text
    for the prompt. Returns a placeholder message if nothing was captured."""
    if not page_context:
        return "(none provided — no live browser capture was run for this generation)"

    lines = []
    for key, label in [("heading", "Heading element"), ("body", "Body text element"), ("button", "Button element")]:
        el = page_context.get(key)
        if el:
            lines.append(
                f"- {label} (<{el.get('tag')}>): font-family: {el.get('font_family')}; "
                f"font-size: {el.get('font_size')}; font-weight: {el.get('font_weight')}; "
                f"color: {el.get('color')}; background-color: {el.get('background_color')}"
            )
    if page_context.get("page_background"):
        lines.append(f"- Page background-color: {page_context['page_background']}")
    if page_context.get("font_face_names"):
        lines.append(f"- Custom @font-face names loaded by the page: {', '.join(page_context['font_face_names'])}")
    if page_context.get("font_links"):
        lines.append(f"- Font stylesheet links: {', '.join(page_context['font_links'])}")

    return "\n".join(lines) if lines else "(browser capture ran but found no usable computed styles)"


def build_stream(reference_url, image_desktop, image_mobile=None, section_name=None, notes=None,
                  video_frames=None, page_context=None):
    """Generator yielding {type: status/done} events.
    image_desktop / image_mobile = {b64, filename, media_type} dicts (image_mobile optional).
    video_frames = list of {b64, media_type} dicts extracted from an optional animation video.
    page_context = optional computed_styles dict from browser_capture.capture_page()."""
    yield {"type": "status", "text": "🌐 Fetching reference page..."}
    title, page_text = fetch_page_text(reference_url)

    status_text = "👁️ Analyzing screenshots + generating section..." if image_mobile else \
                  "👁️ Analyzing screenshot + generating section..."
    yield {"type": "status", "text": status_text}
    prompt = (SECTION_PROMPT_TEMPLATE
              .replace("[REFERENCE_URL]", reference_url)
              .replace("[PAGE_TEXT]", page_text or "(no page text could be fetched)")
              .replace("[MERCHANT_NOTES]", notes.strip() if notes and notes.strip() else "(none provided)")
              .replace("[PAGE_CONTEXT]", _format_page_context(page_context)))

    content = [
        {"type": "text", "text": "DESKTOP SCREENSHOT (reference for the desktop layout):"},
        {"type": "image", "source": {"type": "base64", "media_type": image_desktop["media_type"], "data": image_desktop["b64"]}},
    ]
    if image_mobile:
        content += [
            {"type": "text", "text": "MOBILE SCREENSHOT (ground truth for the exact 749px mobile layout — "
                                      "match this element order/stacking precisely, do not guess):"},
            {"type": "image", "source": {"type": "base64", "media_type": image_mobile["media_type"], "data": image_mobile["b64"]}},
        ]
    if video_frames:
        for i, frame in enumerate(video_frames):
            content += [
                {"type": "text", "text": f"ANIMATION FRAME {i + 1}/{len(video_frames)} (chronological order):"},
                {"type": "image", "source": {"type": "base64", "media_type": frame["media_type"], "data": frame["b64"]}},
            ]
    content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": content}],
            timeout=180.0,
        )
        raw = response.content[0].text
        liquid_code = _extract_liquid(raw)
        liquid_code = _enforce_heading_alignment(liquid_code)
        suggested_name = _extract_schema_name(liquid_code) or section_name or title or "Custom Section"
        yield {"type": "done", "reply": liquid_code, "suggested_name": suggested_name}
    except Exception as e:
        yield {"type": "done", "reply": f"Error: {e}", "suggested_name": section_name or "Custom Section"}


EDIT_PROMPT_TEMPLATE = """ROLE
You are the same expert Shopify Theme developer who generated the custom section below. The
merchant reviewed it and now wants specific corrections applied — this is a targeted revision,
not a from-scratch rebuild.

CURRENT SECTION CODE
```liquid
[CURRENT_CODE]
```

REQUESTED CHANGES (apply these precisely)
[EDIT_INSTRUCTIONS]

RULES FOR THIS REVISION (STRICT)
- Apply ONLY the requested changes above. Preserve every other setting, block, class name, and
  design decision exactly as-is unless a requested change requires touching it.
- Do not regress any of the section's existing quality rules while editing:
  - Every background/text color that should follow the merchant's theme still uses
    rgb(var(--color-background)) / rgb(var(--color-foreground)) — never hardcoded hex.
  - Every distinct accent color (buttons, highlights, icons, bubbles) still has its own
    `"type": "color"` schema setting — never collapse one back into a hardcoded literal.
  - Headings/body text still use font-family: var(--font-heading-family) / var(--font-body-family).
  - Every image_picker still has a clean CSS-only empty-state fallback (never
    `placeholder_svg_tag` chained with `append`).
  - Responsive breakpoints (989px / 749px) still work exactly as before unless the change targets
    them specifically.
- If reference screenshots are attached below, use them only to verify the requested change looks
  right — do not use them to re-derive parts of the design that weren't asked to change.
- If images labeled "REFERENCE IMAGE FOR THIS EDIT" are attached, they specifically illustrate the
  requested change (e.g. a marked-up screenshot, a competitor example, or a photo of what the
  merchant wants instead) — treat them as the primary visual ground truth for whatever aspect of
  the design they show, overriding your own guess about what the instructions mean.
- If images labeled "ANIMATION FRAME" are attached, they show the requested motion/animation in
  chronological order — recreate it with CSS `@keyframes`/`transition` as usual (never JavaScript).

REAL EXTRACTED STYLES (only when provided — ground truth, not a guess)
[PAGE_CONTEXT]

OUTPUT FORMAT (STRICT)
Output ONE thing only: the COMPLETE revised `.liquid` section file (not a diff, not just the
changed lines), wrapped in one ```liquid code fence and nothing else — no explanation before or
after the fence.
"""


def edit_stream(current_code, edit_instructions, reference_url=None, image_desktop=None,
                 image_mobile=None, video_frames=None, extra_images=None, section_name=None,
                 page_context=None):
    """Generator yielding {type: status/done} events. Revises an already-generated section's
    liquid code based on the merchant's follow-up correction instructions.
    extra_images = list of {b64, media_type} dicts attached specifically to illustrate this edit
    (e.g. a marked-up screenshot or a different visual example) — distinct from the original
    desktop/mobile reference screenshots.
    page_context = optional computed_styles dict from browser_capture.capture_page()."""
    yield {"type": "status", "text": "🔧 Applying your changes..."}

    prompt = (EDIT_PROMPT_TEMPLATE
              .replace("[CURRENT_CODE]", current_code)
              .replace("[EDIT_INSTRUCTIONS]", edit_instructions.strip())
              .replace("[PAGE_CONTEXT]", _format_page_context(page_context)))

    content = []
    if image_desktop:
        content += [
            {"type": "text", "text": "DESKTOP SCREENSHOT (for reference while verifying the change):"},
            {"type": "image", "source": {"type": "base64", "media_type": image_desktop["media_type"], "data": image_desktop["b64"]}},
        ]
    if image_mobile:
        content += [
            {"type": "text", "text": "MOBILE SCREENSHOT (for reference while verifying the change):"},
            {"type": "image", "source": {"type": "base64", "media_type": image_mobile["media_type"], "data": image_mobile["b64"]}},
        ]
    if extra_images:
        for i, img in enumerate(extra_images):
            content += [
                {"type": "text", "text": f"REFERENCE IMAGE FOR THIS EDIT {i + 1}/{len(extra_images)} "
                                          "(illustrates the requested change — treat as visual ground truth):"},
                {"type": "image", "source": {"type": "base64", "media_type": img["media_type"], "data": img["b64"]}},
            ]
    if video_frames:
        for i, frame in enumerate(video_frames):
            content += [
                {"type": "text", "text": f"ANIMATION FRAME {i + 1}/{len(video_frames)} (chronological order):"},
                {"type": "image", "source": {"type": "base64", "media_type": frame["media_type"], "data": frame["b64"]}},
            ]
    content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": content}],
            timeout=180.0,
        )
        raw = response.content[0].text
        liquid_code = _extract_liquid(raw)
        liquid_code = _enforce_heading_alignment(liquid_code)
        suggested_name = _extract_schema_name(liquid_code) or section_name or "Custom Section"
        yield {"type": "done", "reply": liquid_code, "suggested_name": suggested_name}
    except Exception as e:
        yield {"type": "done", "reply": f"Error: {e}", "suggested_name": section_name or "Custom Section"}

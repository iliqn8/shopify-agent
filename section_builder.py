import os
import re
import json
import requests
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

WHAT TO BUILD
Analyze ONLY the section visible in the screenshot (ignore header/footer/other sections unless they
are literally inside the screenshot). Recreate it as a brand-new, custom Shopify section.

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
   Only truly decorative accents that must stay constant across every scheme (rare) may use a
   literal color, and only if there is no reasonable alternative.
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


def build_stream(reference_url, image_desktop, image_mobile=None, section_name=None):
    """Generator yielding {type: status/done} events.
    image_desktop / image_mobile = {b64, filename, media_type} dicts (image_mobile optional)."""
    yield {"type": "status", "text": "🌐 Fetching reference page..."}
    title, page_text = fetch_page_text(reference_url)

    status_text = "👁️ Analyzing screenshots + generating section..." if image_mobile else \
                  "👁️ Analyzing screenshot + generating section..."
    yield {"type": "status", "text": status_text}
    prompt = (SECTION_PROMPT_TEMPLATE
              .replace("[REFERENCE_URL]", reference_url)
              .replace("[PAGE_TEXT]", page_text or "(no page text could be fetched)"))

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

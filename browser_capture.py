import base64

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

DESKTOP_VIEWPORT = {"width": 1440, "height": 900}
MOBILE_VIEWPORT = {"width": 390, "height": 844}

_FIND_CONTAINER_JS_BODY = """
  function findContainer(hint) {
    if (!hint) return null;
    const hintLower = hint.toLowerCase();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node, matchEl = null;
    while (node = walker.nextNode()) {
      if (node.textContent && node.textContent.toLowerCase().includes(hintLower)) {
        matchEl = node.parentElement;
        break;
      }
    }
    if (!matchEl) return null;
    let el = matchEl;
    while (el && el !== document.body) {
      const rect = el.getBoundingClientRect();
      if (rect.height >= 150 && rect.height <= 1400 && rect.width >= 300) {
        return el;
      }
      el = el.parentElement;
    }
    return matchEl;
  }
"""

_FIND_CONTAINER_JS = """
(hint) => {
""" + _FIND_CONTAINER_JS_BODY + """
  return findContainer(hint);
}
"""

_LOCATE_AT_POINT_JS = """
([x, y]) => {
  let el = document.elementFromPoint(x, y);
  if (!el) return null;
  while (el && el !== document.body) {
    const rect = el.getBoundingClientRect();
    if (rect.height >= 150 && rect.height <= 1400 && rect.width >= 300) {
      return el;
    }
    el = el.parentElement;
  }
  return el;
}
"""

_STRUCTURE_JS_BODY = """
  function describeContainer(el) {
    const cs = getComputedStyle(el);
    const scrollable = (cs.overflowX === 'auto' || cs.overflowX === 'scroll')
                        && el.scrollWidth > el.clientWidth + 4;
    return {
      display: cs.display,
      flex_direction: cs.flexDirection,
      grid_template_columns: cs.gridTemplateColumns,
      justify_content: cs.justifyContent,
      align_items: cs.alignItems,
      gap: cs.gap,
      overflow_x: cs.overflowX,
      scroll_snap_type: cs.scrollSnapType,
      is_horizontally_scrollable: scrollable,
      scroll_width: el.scrollWidth,
      client_width: el.clientWidth,
    };
  }

  function findScrollContainer(root) {
    const all = [root, ...root.querySelectorAll('*')];
    let best = null;
    let bestCount = 1;
    for (const el of all) {
      const cs = getComputedStyle(el);
      const isRow = cs.display === 'flex' && cs.flexDirection.indexOf('row') === 0;
      const isGrid = cs.display === 'grid';
      if ((isRow || isGrid) && el.children.length > bestCount) {
        best = el;
        bestCount = el.children.length;
      }
    }
    return best || root;
  }

  function describeMedia(root) {
    const out = [];
    root.querySelectorAll('video, img').forEach((el) => {
      if (out.length >= 20) return;
      const rect = el.getBoundingClientRect();
      out.push({
        tag: el.tagName.toLowerCase(),
        src: el.currentSrc || el.getAttribute('src') || null,
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        autoplay: !!el.autoplay,
        muted: !!el.muted,
        loop: !!el.loop,
        controls: !!el.controls,
      });
    });
    return out;
  }

  function describeStructure(section) {
    const scrollContainer = findScrollContainer(section);
    const children = Array.from(scrollContainer.children).slice(0, 20).map((el) => {
      const rect = el.getBoundingClientRect();
      return {
        tag: el.tagName.toLowerCase(),
        classes: (el.className || '').toString().slice(0, 80),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        has_video: !!el.querySelector('video'),
        has_image: !!el.querySelector('img'),
      };
    });
    let html = section.outerHTML || '';
    if (html.length > 6000) html = html.slice(0, 6000) + '... [truncated]';
    return {
      outer_html_excerpt: html,
      container: describeContainer(scrollContainer),
      children_count: scrollContainer.children.length,
      children_summary: children,
      media_elements: describeMedia(section),
    };
  }
"""

_COMPUTED_STYLE_JS = """
(hint) => {
""" + _FIND_CONTAINER_JS_BODY + _STRUCTURE_JS_BODY + """
  function pick(el) {
    if (!el) return null;
    const cs = getComputedStyle(el);
    return {
      tag: el.tagName.toLowerCase(),
      font_family: cs.fontFamily,
      font_size: cs.fontSize,
      font_weight: cs.fontWeight,
      color: cs.color,
      background_color: cs.backgroundColor,
    };
  }

  const section = findContainer(hint);
  const scope = section || document;
  const heading = scope.querySelector('h1, h2, h3');
  const body = scope.querySelector('p');
  const button = scope.querySelector('a.button, button, [class*="btn"], [class*="button"]');

  const fontLinks = Array.from(document.querySelectorAll('link[href*="fonts"]')).map(l => l.href);
  const fontFaceNames = new Set();
  try {
    for (const sheet of document.styleSheets) {
      let rules;
      try { rules = sheet.cssRules; } catch (e) { continue; }
      if (!rules) continue;
      for (const rule of rules) {
        if (rule.constructor && rule.constructor.name === 'CSSFontFaceRule') {
          fontFaceNames.add(rule.style.getPropertyValue('font-family'));
        }
      }
    }
  } catch (e) {}

  return {
    heading: pick(heading),
    body: pick(body),
    button: pick(button),
    page_background: getComputedStyle(document.body).backgroundColor,
    font_links: fontLinks,
    font_face_names: Array.from(fontFaceNames),
    structure: section ? describeStructure(section) : null,
  };
}
"""


def _capture_screenshot(page, section_hint):
    """Screenshot just the matched section container if section_hint locates one,
    otherwise fall back to a full-page screenshot."""
    if section_hint:
        try:
            handle = page.evaluate_handle(_FIND_CONTAINER_JS, section_hint)
            element = handle.as_element()
            if element:
                return element.screenshot(type="jpeg", quality=85), True
        except Exception:
            pass
    return page.screenshot(full_page=True, type="jpeg", quality=85), False


def _decode_image_bytes(raw_bytes):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _locate_template(full_img, template_img, min_confidence=0.45):
    """Multi-scale template matching (the uploaded screenshot may have been captured at a
    different width/zoom than our viewport). Returns {x, y, width, height, confidence} in
    full_img's pixel space, or None if no sufficiently confident match was found."""
    if full_img is None or template_img is None:
        return None
    full_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]
    fh, fw = full_gray.shape[:2]

    best = None  # (score, x, y, w, h)
    for scale in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
        sw, sh = int(tw * scale), int(th * scale)
        if sw < 20 or sh < 20 or sw > fw or sh > fh:
            continue
        resized = cv2.resize(tmpl_gray, (sw, sh))
        result = cv2.matchTemplate(full_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            best = (max_val, max_loc[0], max_loc[1], sw, sh)

    if best is None or best[0] < min_confidence:
        return None
    score, x, y, w, h = best
    return {"x": x, "y": y, "width": w, "height": h, "confidence": float(score)}


def _capture_by_image(page, template_b64):
    """Try to locate the uploaded reference screenshot within the live page via multi-scale
    template matching, then screenshot the containing DOM element and derive a short text
    snippet from it (reused to locate the same section on the mobile viewport).
    Returns (screenshot_bytes_or_None, matched_bool, derived_text_hint_or_None)."""
    try:
        full_bytes = page.screenshot(full_page=True, type="png")
        full_img = _decode_image_bytes(full_bytes)
        template_img = _decode_image_bytes(base64.b64decode(template_b64))
        box = _locate_template(full_img, template_img)
        if not box:
            return None, False, None

        cx = box["x"] + box["width"] // 2
        cy = box["y"] + box["height"] // 2

        page.evaluate("(y) => window.scrollTo(0, Math.max(0, y - 150))", cy)
        scroll_y = page.evaluate("() => window.scrollY")

        handle = page.evaluate_handle(_LOCATE_AT_POINT_JS, [cx, cy - scroll_y])
        element = handle.as_element()
        if not element:
            return None, False, None

        screenshot_bytes = element.screenshot(type="jpeg", quality=85)
        text = (element.text_content() or "").strip()
        text = " ".join(text.split())[:60] or None
        return screenshot_bytes, True, text
    except Exception:
        return None, False, None


def capture_page(url, section_hint=None, template_b64=None, timeout_ms=30000):
    """Launch headless Chromium, load the page, scroll through it, and capture:
    desktop + mobile screenshots, plus real computed styles/fonts for heading/body/button
    representative elements. Never raises — returns a dict with an "error" key on failure
    so callers can fall back to manual screenshots.

    section_hint: an optional short exact phrase of text visible in the target section
    (e.g. a headline). template_b64: an optional base64 screenshot of the desired section —
    if given, takes priority over section_hint and locates the section visually via
    multi-scale template matching against a full-page screenshot, deriving a text hint from
    the matched element for reuse on the mobile viewport. If neither locates a section,
    falls back to a full-page screenshot."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                result = {"error": None}

                # Desktop pass
                page = browser.new_page(viewport=DESKTOP_VIEWPORT,
                                         user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                    "Chrome/120.0 Safari/537.36")
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                _scroll_through(page)

                screenshot_bytes, matched, derived_hint = None, False, None
                if template_b64:
                    screenshot_bytes, matched, derived_hint = _capture_by_image(page, template_b64)

                if screenshot_bytes is None:
                    screenshot_bytes, matched = _capture_screenshot(page, section_hint)

                result["screenshot_desktop_b64"] = base64.b64encode(screenshot_bytes).decode()
                result["section_matched"] = matched
                result["matched_text"] = derived_hint
                result["computed_styles"] = page.evaluate(_COMPUTED_STYLE_JS, derived_hint or section_hint)
                page.close()

                # Mobile pass: reuse the derived/given text hint to locate the same section
                mobile_page = browser.new_page(viewport=MOBILE_VIEWPORT)
                mobile_page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                _scroll_through(mobile_page)
                mobile_screenshot_bytes, _ = _capture_screenshot(mobile_page, derived_hint or section_hint)
                result["screenshot_mobile_b64"] = base64.b64encode(mobile_screenshot_bytes).decode()
                mobile_page.close()

                return result
            finally:
                browser.close()
    except Exception as e:
        return {"error": str(e)}


def _scroll_through(page, step=400, pause_ms=120):
    """Scroll down the full page in steps to trigger lazy-loaded content/animations,
    then back to the top before screenshotting."""
    try:
        page.evaluate(
            """async ({step, pause}) => {
                const height = document.body.scrollHeight;
                for (let y = 0; y < height; y += step) {
                    window.scrollTo(0, y);
                    await new Promise(r => setTimeout(r, pause));
                }
                window.scrollTo(0, 0);
                await new Promise(r => setTimeout(r, pause));
            }""",
            {"step": step, "pause": pause_ms},
        )
    except Exception:
        pass

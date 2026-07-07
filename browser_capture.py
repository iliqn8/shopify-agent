import base64

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

_COMPUTED_STYLE_JS = """
(hint) => {
""" + _FIND_CONTAINER_JS_BODY + """
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

  const scope = findContainer(hint) || document;
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


def capture_page(url, section_hint=None, timeout_ms=30000):
    """Launch headless Chromium, load the page, scroll through it, and capture:
    desktop + mobile screenshots, plus real computed styles/fonts for heading/body/button
    representative elements. Never raises — returns a dict with an "error" key on failure
    so callers can fall back to manual screenshots.

    section_hint: an optional short exact phrase of text visible in the target section
    (e.g. a headline). When provided, the browser locates the smallest reasonably-sized
    container wrapping that text and screenshots ONLY that section instead of the whole
    page. If the phrase isn't found, silently falls back to a full-page screenshot."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                result = {"error": None}

                # Desktop pass: screenshot + computed styles
                page = browser.new_page(viewport=DESKTOP_VIEWPORT,
                                         user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                    "Chrome/120.0 Safari/537.36")
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                _scroll_through(page)
                screenshot_bytes, matched = _capture_screenshot(page, section_hint)
                result["screenshot_desktop_b64"] = base64.b64encode(screenshot_bytes).decode()
                result["section_matched"] = matched
                result["computed_styles"] = page.evaluate(_COMPUTED_STYLE_JS, section_hint)
                page.close()

                # Mobile pass: just the screenshot (styles already captured above)
                mobile_page = browser.new_page(viewport=MOBILE_VIEWPORT)
                mobile_page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                _scroll_through(mobile_page)
                mobile_screenshot_bytes, _ = _capture_screenshot(mobile_page, section_hint)
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

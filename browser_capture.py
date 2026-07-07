import base64

from playwright.sync_api import sync_playwright

DESKTOP_VIEWPORT = {"width": 1440, "height": 900}
MOBILE_VIEWPORT = {"width": 390, "height": 844}

_COMPUTED_STYLE_JS = """
() => {
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

  const heading = document.querySelector('h1, h2, h3');
  const body = document.querySelector('p');
  const button = document.querySelector('a.button, button, [class*="btn"], [class*="button"]');

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


def capture_page(url, timeout_ms=30000):
    """Launch headless Chromium, load the page, scroll through it, and capture:
    desktop + mobile full-page screenshots, plus real computed styles/fonts for
    heading/body/button representative elements. Never raises — returns a dict with
    an "error" key on failure so callers can fall back to manual screenshots."""
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
                result["screenshot_desktop_b64"] = base64.b64encode(
                    page.screenshot(full_page=True, type="jpeg", quality=85)
                ).decode()
                result["computed_styles"] = page.evaluate(_COMPUTED_STYLE_JS)
                page.close()

                # Mobile pass: just the screenshot (styles already captured above)
                mobile_page = browser.new_page(viewport=MOBILE_VIEWPORT)
                mobile_page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                _scroll_through(mobile_page)
                result["screenshot_mobile_b64"] = base64.b64encode(
                    mobile_page.screenshot(full_page=True, type="jpeg", quality=85)
                ).decode()
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

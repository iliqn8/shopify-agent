import re
import shopify_client as sc


def _text_to_html(text):
    lines = text.split('\n')
    html = []
    for line in lines:
        s = line.strip()
        if not s:
            html.append('<br>')
            continue
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        if re.match(r'^(Collapsible Tab|Main Body Section|Top of Page|30-Day Guarantee|FAQ)', s, re.IGNORECASE):
            html.append(f'<h3 style="margin:20px 0 8px">{s}</h3>')
        elif re.match(r'^\d+\.', s):
            html.append(f'<p style="margin:4px 0">{s}</p>')
        elif s.startswith('"') and s.endswith('"'):
            html.append(f'<blockquote style="border-left:3px solid #ccc;padding-left:12px;color:#666;margin:8px 0">{s}</blockquote>')
        elif re.match(r'^—\s', s):
            html.append(f'<p style="color:#888;font-size:13px;margin:2px 0">{s}</p>')
        elif re.match(r'^Q:', s, re.IGNORECASE) or re.match(r'^A:', s, re.IGNORECASE):
            html.append(f'<p style="margin:6px 0"><strong>{s[:2]}</strong>{s[2:]}</p>')
        else:
            html.append(f'<p style="margin:6px 0">{s}</p>')
    return '\n'.join(html)


def parse_output(product_name, generated_text):
    text = generated_text

    # Extract branded title from Section 1
    title = product_name
    m = re.search(r'SECTION 1[^\n]*\n+([^\n]{3,80})', text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().strip('*').strip()
        if candidate and not re.match(r'^(pick|choose|angle|output|this)', candidate, re.IGNORECASE):
            title = candidate

    # Extract price (first $XX.95 pattern)
    price = "34.95"
    m = re.search(r'\$(\d{2,3}\.\d{2})', text)
    if m:
        price = m.group(1)

    # Extract Section 4 content for body_html
    description_text = text
    m = re.search(r'SECTION 4[^\n]*\n+(.*?)(?:SELF-CHECK|---+\s*$|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        description_text = m.group(1).strip()

    return {
        "title": title,
        "price": price,
        "body_html": _text_to_html(description_text),
    }


def publish(product_name, generated_text):
    """Create product + theme template. Returns result dict."""
    parsed = parse_output(product_name, generated_text)

    # Create product
    product = sc.create_product(
        title=parsed["title"],
        body_html=parsed["body_html"],
        price=parsed["price"],
    )
    product_id = str(product["id"])
    handle = product.get("handle", product_id)
    product_url = f"https://{sc.SHOP}/products/{handle}"

    # Build template slug from product name
    slug = re.sub(r'[^a-z0-9]+', '-', product_name.lower()).strip('-')[:40]
    template_suffix = None

    # Create new theme template based on default
    try:
        theme = sc.get_active_theme()
        if theme:
            default = sc.get_theme_file(theme["id"], "templates/product.json")
            template_value = default.get("value") or default.get("attachment") or \
                '{"sections":{"main":{"type":"main-product","disabled":false,"settings":{}}},"order":["main"]}'
            new_key = f"templates/product.{slug}.json"
            sc.update_theme_file(theme["id"], new_key, template_value)
            template_suffix = slug
    except Exception as e:
        print(f"[publisher] Template creation failed: {e}")

    # Assign template to product
    if template_suffix:
        try:
            sc.update_product(product_id, template_suffix=template_suffix)
        except Exception as e:
            print(f"[publisher] Template assign failed: {e}")

    admin_url = f"https://{sc.SHOP.replace('.myshopify.com', '')}.myshopify.com/admin/products/{product_id}"

    return {
        "product_id": product_id,
        "product_url": product_url,
        "admin_url": admin_url,
        "template_suffix": template_suffix,
        "title": parsed["title"],
        "price": parsed["price"],
    }

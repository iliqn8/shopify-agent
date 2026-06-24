import re
import json
import shopify_client as sc


def _bold(text):
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


def _paras(lines):
    return ''.join(f'<p>{_bold(l)}</p>' for l in lines if l.strip())


# ── Parser ─────────────────────────────────────────────────────────────────

def parse_output(product_name, text):
    result = {
        'title': product_name,
        'price': '34.95',
        'colors': {},
        'emoji_bullets': [],
        'how_it_works': [],
        'reviews': [],
        'mb1_headline': '',
        'mb1_paragraphs': [],
        'mb2_headline': '',
        'mb2_blocks': [],
        'mb3_headline': '',
        'mb3_paragraphs': [],
        'guarantee_text': '',
        'faq_items': [],
    }

    # Title
    m = re.search(r'SECTION 1[^\n]*\n+([^\n]{3,80})', text, re.IGNORECASE)
    if m:
        c = m.group(1).strip().strip('*').strip()
        if c and not re.match(r'^(pick|choose|angle|output|this|do not)', c, re.IGNORECASE):
            result['title'] = c

    # Price
    m = re.search(r'\$(\d{2,3}\.\d{2})', text)
    if m:
        result['price'] = m.group(1)

    # Colors
    for label, key in [('Background color', 'bg'), ('Text color', 'text'),
                        ('Accent 1', 'accent1'), ('Accent 2', 'accent2')]:
        m = re.search(label + r'[^#]*#([0-9A-Fa-f]{6})', text, re.IGNORECASE)
        if m:
            result['colors'][key] = '#' + m.group(1)

    # Top of Page emoji bullets
    m = re.search(r'Top of Page[^\n]*\n+(.*?)(?:Collapsible Tab|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        result['emoji_bullets'] = lines[:3]

    # How It Works
    m = re.search(r'Collapsible Tab[^\n]*How It Works[^\n]*\n+(.*?)(?:Collapsible Tab|Main Body|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        result['how_it_works'] = lines[:3]

    # Reviews
    m = re.search(r'Collapsible Tab[^\n]*Review[^\n]*\n+(.*?)(?:Main Body|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        blocks = re.findall(r'"([^"]+)"\s*\n+[—\-]\s*([^\n]+)', m.group(1))
        result['reviews'] = [{'text': q.strip(), 'author': a.strip()} for q, a in blocks[:3]]

    # Main Body Section 1
    m = re.search(r'Main Body Section 1[^\n]*\n+(.*?)(?:Main Body Section 2|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        if lines:
            result['mb1_headline'] = lines[0]
            result['mb1_paragraphs'] = lines[1:]

    # Main Body Section 2
    m = re.search(r'Main Body Section 2[^\n]*\n+(.*?)(?:Main Body Section 3|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        if lines:
            result['mb2_headline'] = lines[0]
            i = 1
            while i < len(lines) and len(result['mb2_blocks']) < 4:
                # Emoji line: short, not starting with ** or letter
                line = lines[i]
                is_emoji = len(line) <= 4 and not line.startswith('**') and not line[0].isascii() or (len(line) <= 3)
                if is_emoji or re.match(r'^[\U0001F000-\U0001FFFF]|^[\U00002600-\U000027BF]', line):
                    emoji = line
                    title_l = lines[i+1].strip('*').strip() if i+1 < len(lines) else ''
                    desc_l = lines[i+2] if i+2 < len(lines) else ''
                    result['mb2_blocks'].append({'emoji': emoji, 'title': title_l, 'desc': desc_l})
                    i += 3
                else:
                    i += 1

    # Main Body Section 3
    m = re.search(r'Main Body Section 3[^\n]*\n+(.*?)(?:30-Day|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        if lines:
            result['mb3_headline'] = lines[0]
            result['mb3_paragraphs'] = lines[1:]

    # 30-Day Guarantee
    m = re.search(r'30-Day Guarantee[^\n]*\n+(.*?)(?:FAQ|$)', text, re.IGNORECASE | re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        result['guarantee_text'] = ' '.join(lines)

    # FAQ
    m = re.search(r'\bFAQ\b[^\n]*\n+(.*?)$', text, re.IGNORECASE | re.DOTALL)
    if m:
        faq_lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
        i = 0
        while i < len(faq_lines) and len(result['faq_items']) < 4:
            q = faq_lines[i]
            if '?' in q:
                ans = []
                j = i + 1
                while j < len(faq_lines) and '?' not in faq_lines[j]:
                    ans.append(faq_lines[j])
                    j += 1
                result['faq_items'].append({'q': q, 'a': ' '.join(ans)})
                i = j
            else:
                i += 1

    return result


# ── Template builder ───────────────────────────────────────────────────────

def fill_template(template_json_str, parsed):
    tmpl = json.loads(template_json_str)
    colors = parsed.get('colors', {})

    # ── main section blocks ──
    main_blocks = tmpl['sections']['main']['blocks']

    # 3 emoji bullets
    if parsed['emoji_bullets']:
        html = ''.join(f'<p>{_bold(b)}</p>' for b in parsed['emoji_bullets'])
        if 'emoji_benefits_xFGiTn' in main_blocks:
            main_blocks['emoji_benefits_xFGiTn']['settings']['benefits'] = html

    # How It Works tab
    if parsed['how_it_works'] and 'collapsible_tab_6mMkwr' in main_blocks:
        steps = ''.join(f'<p>{i+1}. {_bold(s)}</p>' for i, s in enumerate(parsed['how_it_works']))
        main_blocks['collapsible_tab_6mMkwr']['settings']['heading'] = 'How It Works'
        main_blocks['collapsible_tab_6mMkwr']['settings']['content'] = steps

    # Reviews in product info block
    for i, rev in enumerate(parsed['reviews'][:3], 1):
        if 'reviews_wbqVgr' in main_blocks:
            main_blocks['reviews_wbqVgr']['settings'][f'text_{i}'] = f'<p><em>"{rev["text"]}"</em></p>'
            main_blocks['reviews_wbqVgr']['settings'][f'author_{i}'] = rev['author']

    # Button colors
    if colors.get('accent1'):
        if 'buy_buttons' in main_blocks:
            main_blocks['buy_buttons']['settings']['enable_custom_color'] = True
            main_blocks['buy_buttons']['settings']['custom_color'] = colors['accent1']
        if 'sticky_atc_xbBkLM' in main_blocks:
            main_blocks['sticky_atc_xbBkLM']['settings']['enable_custom_btn_color'] = True
            main_blocks['sticky_atc_xbBkLM']['settings']['custom_btn_color'] = colors['accent1']

    # ── image_with_text_6NJQ98 — Main Body Section 1 ──
    s1 = tmpl['sections'].get('image_with_text_6NJQ98', {})
    if s1 and parsed['mb1_headline']:
        b = s1.get('blocks', {})
        if 'heading_hJVTy3' in b:
            b['heading_hJVTy3']['settings']['heading'] = _bold(parsed['mb1_headline'])
        if 'text_apQhMK' in b and parsed['mb1_paragraphs']:
            b['text_apQhMK']['settings']['text'] = _paras(parsed['mb1_paragraphs'])

    # ── benefit_icons_image_eNxPJQ — Main Body Section 2 ──
    s2 = tmpl['sections'].get('benefit_icons_image_eNxPJQ', {})
    if s2:
        if parsed['mb2_headline']:
            s2['settings']['headline'] = _bold(parsed['mb2_headline'])
        if colors.get('bg'):
            s2['settings']['bg_color'] = colors['bg']
        if colors.get('text'):
            s2['settings']['headline_color'] = colors['text']
            s2['settings']['subhead_color'] = colors['text']
        if colors.get('accent1'):
            s2['settings']['icon_color'] = colors['accent1']
        # Fill 4 benefit blocks
        b = s2.get('blocks', {})
        bid_list = list(b.keys())
        for idx, block_data in enumerate(parsed['mb2_blocks'][:4]):
            if idx < len(bid_list):
                bid = bid_list[idx]
                b[bid]['settings']['icon_type'] = 'emoji'
                b[bid]['settings']['emoji_text'] = block_data.get('emoji', '✨')
                b[bid]['settings']['title'] = block_data.get('title', '')
                b[bid]['settings']['text'] = block_data.get('desc', '')

    # ── image_with_text_8wqzxh — Main Body Section 3 ──
    s3 = tmpl['sections'].get('image_with_text_8wqzxh', {})
    if s3 and parsed['mb3_headline']:
        b = s3.get('blocks', {})
        if 'heading_MGgztr' in b:
            b['heading_MGgztr']['settings']['heading'] = _bold(parsed['mb3_headline'])
        if 'text_KpKUUF' in b and parsed['mb3_paragraphs']:
            b['text_KpKUUF']['settings']['text'] = _paras(parsed['mb3_paragraphs'])

    # ── rich_text_d7MAiq — 30-Day Guarantee ──
    rich = tmpl['sections'].get('rich_text_d7MAiq', {})
    if rich and parsed['guarantee_text']:
        rb = rich.get('blocks', {})
        if 'text_Vrfa8P' in rb:
            rb['text_Vrfa8P']['settings']['text'] = f'<p>{_bold(parsed["guarantee_text"])}</p>'
    if colors.get('accent1') and rich:
        rich['settings']['custom_colors_solid_button_background'] = colors['accent1']

    # ── ds_testimonials_i86BLn — Customer Reviews ──
    testi = tmpl['sections'].get('ds_testimonials_i86BLn', {})
    if testi and parsed['reviews']:
        tb = testi.get('blocks', {})
        tb_ids = list(tb.keys())
        for idx, rev in enumerate(parsed['reviews'][:3]):
            if idx < len(tb_ids):
                bid = tb_ids[idx]
                tb[bid]['settings']['text'] = f'<p><em>"{rev["text"]}"</em></p>'
                tb[bid]['settings']['author'] = rev['author']
                tb[bid]['settings']['title'] = ' '.join(rev['text'].split()[:4]) + '...'

    # ── collapsible_content_ea4B3M — FAQ ──
    faq_sec = tmpl['sections'].get('collapsible_content_ea4B3M', {})
    if faq_sec and parsed['faq_items']:
        fb = faq_sec.get('blocks', {})
        block_order = faq_sec.get('block_order', [])
        # Use the first 4 Question slots
        q_slots = [bid for bid in block_order
                   if fb.get(bid, {}).get('settings', {}).get('heading', '').startswith('Question')]
        for idx, faq in enumerate(parsed['faq_items'][:4]):
            if idx < len(q_slots):
                bid = q_slots[idx]
                fb[bid]['settings']['heading'] = faq['q']
                fb[bid]['settings']['row_content'] = f'<p>{_bold(faq["a"])}</p>'

    return json.dumps(tmpl)


# ── Theme color updater ────────────────────────────────────────────────────

def apply_theme_colors(tid, colors):
    if not colors:
        return
    raw = sc.get_theme_file(tid, 'config/settings_data.json')
    data = json.loads(raw.get('value') or raw.get('attachment') or '{}')
    cur = data.get('current', {})

    if colors.get('bg'):
        cur['colors_background_1'] = colors['bg']
    if colors.get('text'):
        cur['colors_text'] = colors['text']
    if colors.get('accent1'):
        cur['colors_accent_1'] = colors['accent1']
    if colors.get('accent2'):
        cur['colors_accent_2'] = colors['accent2']

    # Also update scheme-1 (the primary scheme used by most sections)
    s1 = cur.get('color_schemes', {}).get('scheme-1', {}).get('settings', {})
    if s1:
        if colors.get('bg'):
            s1['background'] = colors['bg']
        if colors.get('text'):
            s1['text'] = colors['text']
            s1['secondary_button_label'] = colors['text']
        if colors.get('accent1'):
            s1['button'] = colors['accent1']

    data['current'] = cur
    sc.update_theme_file(tid, 'config/settings_data.json', json.dumps(data))


# ── Main publish function ──────────────────────────────────────────────────

def publish(product_name, generated_text):
    parsed = parse_output(product_name, generated_text)

    # Create Shopify product
    product = sc.create_product(
        title=parsed['title'],
        body_html=_paras(parsed['mb1_paragraphs']) or '',
        price=parsed['price'],
    )
    product_id = str(product['id'])
    handle = product.get('handle', product_id)
    product_url = f'https://{sc.SHOP}/products/{handle}'
    admin_url = f'https://{sc.SHOP}/admin/products/{product_id}'

    # Build template slug
    slug = re.sub(r'[^a-z0-9]+', '-', product_name.lower()).strip('-')[:40]
    template_suffix = None

    # Create filled template + apply colors
    try:
        theme = sc.get_active_theme()
        if theme:
            tid = theme['id']
            default = sc.get_theme_file(tid, 'templates/product.json')
            base_json = default.get('value') or default.get('attachment') or '{}'
            filled_json = fill_template(base_json, parsed)
            new_key = f'templates/product.{slug}.json'
            sc.update_theme_file(tid, new_key, filled_json)
            template_suffix = slug
            # Apply brand colors to theme settings
            if parsed.get('colors'):
                apply_theme_colors(tid, parsed['colors'])
    except Exception as e:
        print(f'[publisher] Template error: {e}')

    # Assign template to product
    if template_suffix:
        try:
            sc.update_product(product_id, template_suffix=template_suffix)
        except Exception as e:
            print(f'[publisher] Assign error: {e}')

    return {
        'product_id': product_id,
        'product_url': product_url,
        'admin_url': admin_url,
        'template_suffix': template_suffix,
        'title': parsed['title'],
        'price': parsed['price'],
    }

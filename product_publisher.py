import re
import json
import shopify_client as sc


# The page copy fills blocks that exist in both product templates, but only the
# gummies one carries the features grid and the comparison table. Copying from
# the bare product.json produced a page with nowhere to put two of the sections
# the prompt writes.
BASE_TEMPLATE = 'templates/product.gummies.json'
FALLBACK_TEMPLATE = 'templates/product.json'


def _bold(text):
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


def _paras(lines):
    return ''.join(f'<p>{_bold(l)}</p>' for l in lines if l.strip())


# ── Parser ─────────────────────────────────────────────────────────────────

# Every heading the generated output can put between two blocks. Each block
# below stops at whichever of these comes first, instead of trusting that the
# next thing on the page is the one block it expects — the comparison table
# sits between Main Body 3 and the guarantee, and used to be swallowed whole.
# A response may write its headings plainly or as markdown, and the two have
# alternated between runs. "## Main Body Section 2" is the same heading as
# "MAIN BODY SECTION 2"; missing that is how one section swallows the next.
_LEAD = r"(?:[#*\-\s]{0,8})"
_STOP = (r"(?:\n" + _LEAD + r"(?:SECTION\s+\d|TOP OF PAGE|COLLAPSIBLE TAB|MAIN BODY SECTION\s*\d"
         r"|COMPARISON TABLE|FEATURES GRID|ROWS\b|30\s*-?\s*DAY GUARANTEE|FAQ\b"
         r"|SELF\s*-?\s*CHECK|CRITICAL RULES|PALETTE\b|CHECKS\b|COLOUR SCHEMES"
         r"|COLOR SCHEMES|GLOBAL COLOURS?|SECTION SCHEMES|SOURCE SWITCHES"
         r"|FIELDS, BY SECTION|Word count:|Failures used:|Insider phrases:"
         r"|Competitor tick test:)|$)")

# Markdown heading markers, and the stray bold a headline sometimes arrives in.
_MARKS = re.compile(r"^\s*(?:#{1,6}\s*|[*\-]\s+)")


def _clean_line(line):
    line = _MARKS.sub("", line.strip())
    return line.strip()


def _block(text, start):
    """The lines under `start`, up to the next heading of any kind."""
    m = re.search(start + r"[^\n]*\n+(.*?)" + _STOP, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    return [c for c in (_clean_line(l) for l in m.group(1).split(chr(10))) if c]


def parse_output(product_name, text):
    result = {
        'title': product_name,
        'price': '34.95',
        'price_source': '',
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

    # Price. The first dollar figure on the page is the cost of goods, not the
    # price — reading it as the price creates the product at a loss. Look for
    # the line that names the selling price, and only guess if it is absent.
    price_patterns = [
        (r'recommended\s+selling\s+price[^\d$]{0,20}\$?\s*(\d{1,4}\.\d{2})', 'recommended selling price'),
        (r'selling\s+price[^\d$]{0,20}\$?\s*(\d{1,4}\.\d{2})', 'selling price'),
    ]
    for pattern, source in price_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result['price'] = m.group(1)
            result['price_source'] = source
            break
    else:
        # Nothing named it, so this is a guess and gets flagged as one. Every
        # price on the ladder ends in .95; take the lowest, because the figure
        # above it is usually the compare-at price, not what the customer pays.
        ladder = re.findall(r'\$(\d{1,4}\.95)\b', text)
        if ladder:
            result['price'] = min(ladder, key=float)
            result['price_source'] = 'guessed — no "selling price" line in the output'
        else:
            result['price_source'] = 'default — no price found in the output'

    # Colors
    # The PALETTE block names the roles the theme is keyed to, so read that
    # first. Section 3 prints the same colours under a designer's names, and
    # those change whenever the brief does; the role names do not.
    for role, key in [('BG', 'bg'), ('INK', 'text'), ('ACCENT', 'accent1'),
                      ('ACCENT_SOFT', 'accent2'), ('CONTRAST', 'contrast')]:
        m = re.search(r'^\s*' + role + r'\s*=\s*#([0-9A-Fa-f]{6})', text, re.M)
        if m:
            result['colors'][key] = '#' + m.group(1)
    # Designer names and older output, as a fallback.
    for label, key in [('Background color', 'bg'), ('Background', 'bg'),
                       ('Text color', 'text'), ('Heading text', 'text'),
                       ('Accent 1', 'accent1'), ('Primary', 'accent1'),
                       ('Accent 2', 'accent2'), ('Secondary', 'accent2'),
                       ('Accent / CTA', 'contrast'), ('Accent/CTA', 'contrast')]:
        if result['colors'].get(key):
            continue
        m = re.search(label + r'[^#\n]{0,24}#([0-9A-Fa-f]{6})', text, re.IGNORECASE)
        if m:
            result['colors'][key] = '#' + m.group(1)

    # Top of Page emoji bullets
    result['emoji_bullets'] = _block(text, r'Top of Page')[:3]

    # How It Works
    result['how_it_works'] = _block(text, r'Collapsible Tab[^\n]*How It Works')[:3]

    # Reviews
    review_lines = _block(text, r'Collapsible Tab[^\n]*Review')
    blocks = re.findall(r'"([^"]+)"\s*\n+[\u2014\-]\s*([^\n]+)', chr(10).join(review_lines))
    result['reviews'] = [{'text': q.strip(), 'author': a.strip()} for q, a in blocks[:3]]

    # Main Body Section 1
    lines = _block(text, r'Main Body Section 1')
    if lines:
        result['mb1_headline'] = lines[0]
        result['mb1_paragraphs'] = lines[1:]

    # Main Body Section 2
    lines = _block(text, r'Main Body Section 2')
    if True:
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
    lines = _block(text, r'Main Body Section 3')
    if lines:
        result['mb3_headline'] = lines[0]
        result['mb3_paragraphs'] = lines[1:]

    # 30-Day Guarantee
    result['guarantee_text'] = ' '.join(_block(text, r'30\s*-?\s*Day Guarantee'))

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
    # The buy button reads its colour from Scheme 4 now, so it is left alone.
    # The sticky bar has no such switch and has to be told the same colour by
    # hand, or the two disagree with each other on every scroll.
    dark = colors.get('contrast') or colors.get('accent1')
    if dark and 'sticky_atc_xbBkLM' in main_blocks:
        main_blocks['sticky_atc_xbBkLM']['settings']['enable_custom_btn_color'] = True
        main_blocks['sticky_atc_xbBkLM']['settings']['custom_btn_color'] = dark

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
    # rich-text's solid button follows Scheme 4 through its own source switch.

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


# ── Main publish function ──────────────────────────────────────────────────

# ── The guard ──────────────────────────────────────────────────────────────
# Three separate times a change in how the response was formatted let one block
# run into the next, and the scaffolding of the prompt itself reached a live
# page: a comparison table printed beside a product photo, a FAQ inside the
# guarantee, "###" in a headline. Each time the parser was taught the new shape.
#
# This checks the result instead of the input, so the next format nobody
# predicted is caught here rather than by looking at the published page. If it
# fires, nothing is created — a page that has to be repaired by hand costs more
# than a generation that has to be run again.

# Things that are never product copy. A hit means a block ran past its end.
_SCAFFOLD = [
    (r"(?:^|\s)#{2,6}\s+\S",                       "a markdown heading"),
    (r"\b(?:featured_title|other_title|heading|subheading)\s*=",
                                                "a field assignment"),
    (r"^\s*\d+\s+feature\s*=",                 "a comparison table row"),
    (r"\bROWS\b",                              "the ROWS marker"),
    (r"\bWord count\s*:",                      "the word-count line"),
    (r"\bFailures used\s*:",                   "the failures line"),
    (r"\bInsider phrases\s*:",                 "the insider-phrases line"),
    (r"\bCompetitor tick test\s*:",            "the tick-test line"),
    (r"\bMain Body Section\s*\d",              "another section's heading"),
    (r"\bCOMPARISON TABLE\b",                  "the comparison table heading"),
    (r"\bTop of Page\b",                       "the top-of-page heading"),
    (r"\bCollapsible Tab\b",                   "a collapsible-tab heading"),
    (r"\b30\s*-?\s*Day Guarantee\b",           "the guarantee heading"),
    (r"\bSECTION\s+\d\b",                     "a SECTION heading"),
    (r"#[0-9A-Fa-f]{6}\b",                      "a hex colour"),
]

# Roughly twice the longest of these that has ever read well, so an overrun is
# caught even when it carries no scaffolding at all.
_LIMITS = {
    "Main Body 1": 1100, "Main Body 3": 1100, "the guarantee": 400,
    "a FAQ answer": 700, "a benefit description": 260, "a review": 500,
}


def check_parsed(parsed):
    """Everything wrong with a parse, in the order a person would read it."""
    problems = []

    def look(label, value, limit_key=None):
        text = value if isinstance(value, str) else chr(10).join(value or [])
        if not text:
            return
        for pattern, what in _SCAFFOLD:
            if re.search(pattern, text, re.IGNORECASE | re.M):
                problems.append(
                    "%s contains %s, so it ran past where it should have stopped."
                    % (label, what))
                break
        cap = _LIMITS.get(limit_key or label)
        if cap and len(text) > cap:
            problems.append("%s is %d characters, and nothing here should pass %d."
                            % (label, len(text), cap))

    look("Main Body 1", [parsed["mb1_headline"]] + parsed["mb1_paragraphs"], "Main Body 1")
    look("Main Body 2", [parsed["mb2_headline"]])
    look("Main Body 3", [parsed["mb3_headline"]] + parsed["mb3_paragraphs"], "Main Body 3")
    look("the guarantee", parsed["guarantee_text"], "the guarantee")
    look("the top-of-page bullets", parsed["emoji_bullets"])
    look("How It Works", parsed["how_it_works"])
    for i, blk in enumerate(parsed["mb2_blocks"], 1):
        look("benefit %d" % i, [blk.get("title", ""), blk.get("desc", "")],
             "a benefit description")
    for i, rev in enumerate(parsed["reviews"], 1):
        look("review %d" % i, rev.get("text", ""), "a review")
    for i, item in enumerate(parsed["faq_items"], 1):
        look("FAQ %d" % i, item.get("a", ""), "a FAQ answer")

    # Empty is its own kind of wrong: it means a heading was not recognised.
    for label, value in (("Main Body 1", parsed["mb1_paragraphs"]),
                         ("Main Body 3", parsed["mb3_paragraphs"]),
                         ("the FAQ", parsed["faq_items"]),
                         ("the benefit blocks", parsed["mb2_blocks"])):
        if not value:
            problems.append("%s came out empty, so its heading was not recognised."
                            % label)
    return problems


class ParseProblem(Exception):
    """Raised instead of publishing something that would need repairing."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__("The generated output did not parse cleanly.")


def publish(product_name, generated_text):
    parsed = parse_output(product_name, generated_text)
    problems = check_parsed(parsed)
    if problems:
        raise ParseProblem(problems)

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
            try:
                default = sc.get_theme_file(tid, BASE_TEMPLATE)
            except Exception:
                default = sc.get_theme_file(tid, FALLBACK_TEMPLATE)
            base_json = default.get('value') or default.get('attachment') or '{}'
            filled_json = fill_template(base_json, parsed)
            new_key = f'templates/product.{slug}.json'
            sc.update_theme_file(tid, new_key, filled_json)
            template_suffix = slug
            # Global theme colours are deliberately not touched here. They
            # affect every page in the store, so they belong to the Write into
            # the theme step, which shows a diff and downloads a backup first.
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
        'price_source': parsed.get('price_source', ''),
        'base_template': BASE_TEMPLATE,
    }

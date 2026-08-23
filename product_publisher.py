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


def _plain(text):
    """For fields that take text, not HTML — the ** would show up as itself."""
    return re.sub(r'\*\*(.+?)\*\*', r'\1', text or '').strip()


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


# Inside a block, a markdown heading is the next block starting. Whatever the
# heading says, the lines under it are not this block's.
_HEADING = re.compile(r"^\s*#{1,6}\s+\S")

# Lines trimmed on the way through, so the result can say what it dropped
# rather than quietly shortening the page.
_trimmed = []


def _block(text, start, label=None):
    """The lines under `start`, up to the next heading of any kind.

    The stop list catches the headings it knows. This also cuts at anything
    that reads like scaffolding, so a heading written in a shape nobody has
    seen yet costs a trimmed line instead of a broken page.
    """
    m = re.search(start + r"[^\n]*\n+(.*?)" + _STOP, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    out = []
    for line in m.group(1).split(chr(10)):
        if not line.strip():
            continue
        # The first line is this block's own headline, and it often arrives
        # as "### ...". Only a heading after that is the next block starting.
        if out and (_HEADING.match(line) or _is_scaffold(line)):
            _trimmed.append((label or start, line.strip()[:70], len(out)))
            break
        cleaned = _clean_line(line)
        if cleaned:
            out.append(cleaned)
    return out


def parse_output(product_name, text):
    del _trimmed[:]
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
    result['emoji_bullets'] = _block(text, r'Top of Page', 'the top-of-page bullets')[:3]

    # How It Works
    result['how_it_works'] = _block(text, r'Collapsible Tab[^\n]*How It Works', 'How It Works')[:3]

    # Reviews
    review_lines = _block(text, r'Collapsible Tab[^\n]*Review', 'the reviews')
    blocks = re.findall(r'"([^"]+)"\s*\n+[\u2014\-]\s*([^\n]+)', chr(10).join(review_lines))
    result['reviews'] = [{'text': q.strip(), 'author': a.strip()} for q, a in blocks[:3]]

    # Main Body Section 1
    lines = _block(text, r'Main Body Section 1', 'Main Body 1')
    if lines:
        result['mb1_headline'] = lines[0]
        result['mb1_paragraphs'] = lines[1:]

    # Main Body Section 2
    lines = _block(text, r'Main Body Section 2', 'Main Body 2')
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
    lines = _block(text, r'Main Body Section 3', 'Main Body 3')
    if lines:
        result['mb3_headline'] = lines[0]
        result['mb3_paragraphs'] = lines[1:]

    # 30-Day Guarantee
    result['guarantee_text'] = ' '.join(_block(text, r'30\s*-?\s*Day Guarantee', 'the guarantee'))

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

    result['trimmed'] = ['%s stopped at "%s"' % (label, snippet)
                         for label, snippet, _kept in _trimmed]
    return result


# ── Template builder ───────────────────────────────────────────────────────

def _bullet(text):
    """A tick-list line: no emoji, no asterisks — the list draws its own icon."""
    text = re.sub(r'^\s*[^\w\s(&"\'\u2018\u201c-]+\s*', '', text or '')
    return _plain(text)


def fill_template(template_json_str, parsed):
    """Fill the copy, and return the filled template with a record of every
    product-specific field written, as (label, section, block, key)."""
    tmpl = json.loads(template_json_str)
    colors = parsed.get('colors', {})
    written = []

    def put(sid, bid, key, value, label=None):
        """Write one setting.

        A label marks it as copy that has to end up different from the template
        this was copied from. The expectation is recorded whether or not there
        is anything to write, because a field nobody wrote is exactly how a page
        ends up talking about the previous product.
        """
        sec = tmpl['sections'].get(sid)
        if not sec:
            return
        if bid is None:
            store = sec.setdefault('settings', {})
        else:
            blocks = sec.get('blocks') or {}
            if bid not in blocks:
                return
            store = blocks[bid].setdefault('settings', {})
        if label and (label, sid, bid, key) not in written:
            written.append((label, sid, bid, key))
        if value is None or value == '' or value == []:
            return
        store[key] = value

    # ── the three ticked bullets under the price ──
    put('main', 'emoji_benefits_xFGiTn', 'benefits',
        ''.join(f'<p>{_bold(b)}</p>' for b in parsed['emoji_bullets']),
        'the bullets under the price')
    if True:
        # The same three again, in a hand-written list that carries its own
        # tick icons. It was never filled, so every page has shown the base
        # template's bullets — pH balance, on a page about vein patches.
        blocks = tmpl['sections'].get('main', {}).get('blocks', {})
        for bid, blk in blocks.items():
            html = (blk.get('settings') or {}).get('custom_liquid', '')
            if blk.get('type') != 'custom_liquid' or 'product-benefits' not in html:
                continue
            lines = [_bullet(b) for b in parsed['emoji_bullets'][:3]]
            n = [0]

            def swap(m):
                i = n[0]
                n[0] += 1
                return '<span>%s</span>' % lines[i] if i < len(lines) else m.group(0)

            put('main', bid, 'custom_liquid',
                re.sub(r'<span>.*?</span>', swap, html, flags=re.S) if lines else '',
                'the ticked benefit list')

    # ── How It Works ──
    if parsed['how_it_works']:
        put('main', 'collapsible_tab_6mMkwr', 'heading', 'How It Works')
    put('main', 'collapsible_tab_6mMkwr', 'content',
        ''.join(f'<p>{i+1}. {_bold(s)}</p>'
                for i, s in enumerate(parsed['how_it_works'])),
        'the How It Works tab')

    # ── the three short reviews beside the price ──
    revs = parsed['reviews'][:3]
    for i in (1, 2, 3):
        rev = revs[i - 1] if i <= len(revs) else None
        put('main', 'reviews_wbqVgr', f'text_{i}',
            f'<p><em>"{rev["text"]}"</em></p>' if rev else '',
            f'review {i} beside the price')
        if rev:
            put('main', 'reviews_wbqVgr', f'author_{i}', rev['author'])

    # The buy button reads its colour from Scheme 4. The sticky bar has no such
    # switch and has to be told the same colour by hand.
    dark = colors.get('contrast') or colors.get('accent1')
    if dark:
        put('main', 'sticky_atc_xbBkLM', 'enable_custom_btn_color', True)
        put('main', 'sticky_atc_xbBkLM', 'custom_btn_color', dark)

    # ── Main Body 1 ──
    put('image_with_text_6NJQ98', 'heading_hJVTy3', 'heading',
        _bold(parsed['mb1_headline']), 'Main Body 1')
    put('image_with_text_6NJQ98', 'text_apQhMK', 'text',
        _paras(parsed['mb1_paragraphs']), 'Main Body 1')

    # ── Main Body 2 ──
    put('benefit_icons_image_eNxPJQ', None, 'headline',
        _bold(parsed['mb2_headline']), 'Main Body 2')
    if colors.get('bg'):
        put('benefit_icons_image_eNxPJQ', None, 'bg_color', colors['bg'])
    if colors.get('text'):
        put('benefit_icons_image_eNxPJQ', None, 'headline_color', colors['text'])
        put('benefit_icons_image_eNxPJQ', None, 'subhead_color', colors['text'])
    if colors.get('accent1'):
        put('benefit_icons_image_eNxPJQ', None, 'icon_color', colors['accent1'])
    s2 = tmpl['sections'].get('benefit_icons_image_eNxPJQ', {})
    bids = list((s2.get('blocks') or {}).keys())
    for idx in range(min(4, len(bids))):
        blk = parsed['mb2_blocks'][idx] if idx < len(parsed['mb2_blocks']) else {}
        if blk:
            put('benefit_icons_image_eNxPJQ', bids[idx], 'icon_type', 'emoji')
            put('benefit_icons_image_eNxPJQ', bids[idx], 'emoji_text',
                blk.get('emoji', '\u2728'))
        put('benefit_icons_image_eNxPJQ', bids[idx], 'title',
            blk.get('title', ''), 'benefit %d' % (idx + 1))
        put('benefit_icons_image_eNxPJQ', bids[idx], 'text',
            blk.get('desc', ''), 'benefit %d' % (idx + 1))

    # ── Main Body 3 ──
    put('image_with_text_8wqzxh', 'heading_MGgztr', 'heading',
        _bold(parsed['mb3_headline']), 'Main Body 3')
    put('image_with_text_8wqzxh', 'text_KpKUUF', 'text',
        _paras(parsed['mb3_paragraphs']), 'Main Body 3')

    # ── the guarantee ──
    put('rich_text_d7MAiq', 'text_Vrfa8P', 'text',
        f'<p>{_bold(parsed["guarantee_text"])}</p>' if parsed['guarantee_text'] else '',
        'the guarantee')

    # ── the reviews carousel ──
    # Never filled before, so every page carried the same five reviews about a
    # ring. Three real ones is better than five borrowed; the spare slots go.
    hc = [sid for sid, s in tmpl['sections'].items()
          if s.get('type') == 'custom-happy-customers-carousel']
    for sid in hc:
        sec = tmpl['sections'][sid]
        order = sec.get('block_order') or list((sec.get('blocks') or {}).keys())
        used = 0
        if order:
            put(sid, order[0], 'text', '', 'reviews carousel')
        for idx, rev in enumerate(parsed['reviews'][:len(order)]):
            bid = order[idx]
            put(sid, bid, 'name', rev['author'])
            put(sid, bid, 'title',
                _plain(' '.join(rev['text'].split()[:4])).rstrip('.,') + '...')
            put(sid, bid, 'text', rev['text'], 'reviews carousel')
            used = idx + 1
        if used:
            for bid in order[used:]:
                (sec.get('blocks') or {}).pop(bid, None)
            sec['block_order'] = order[:used]

    # ── the FAQ ──
    # This used to look for rows still headed "Question 1", which is what an
    # empty template carries. The base has real questions in it, so nothing
    # matched and every page kept the base's FAQ. Rows are taken in order now;
    # the ones past the answers we have are left alone, which is where the
    # shipping and contact rows live.
    faq = tmpl['sections'].get('collapsible_content_ea4B3M', {})
    slots = faq.get('block_order') or list((faq.get('blocks') or {}).keys())
    for idx in range(min(4, len(slots))):
        item = parsed['faq_items'][idx] if idx < len(parsed['faq_items']) else {}
        put('collapsible_content_ea4B3M', slots[idx], 'heading',
            _plain(item.get('q', '')), 'FAQ %d' % (idx + 1))
        put('collapsible_content_ea4B3M', slots[idx], 'row_content',
            f'<p>{_bold(item["a"])}</p>' if item.get('a') else '',
            'FAQ %d' % (idx + 1))

    return json.dumps(tmpl), written


def unwritten(base_json_str, filled_json_str, written):
    """Copy that was written and still says exactly what the base said.

    Either the response had nothing for it or the write missed, and the page
    would go live talking about whatever product the template was copied from.
    """
    base = json.loads(base_json_str).get('sections', {})
    filled = json.loads(filled_json_str).get('sections', {})

    def read(tree, sid, bid, key):
        sec = tree.get(sid) or {}
        store = (sec.get('settings') or {}) if bid is None else \
                ((sec.get('blocks') or {}).get(bid) or {}).get('settings') or {}
        return store.get(key)

    stale = []
    for label, sid, bid, key in written:
        if read(base, sid, bid, key) == read(filled, sid, bid, key):
            if label not in stale:
                stale.append(label)
    return stale


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


def _is_scaffold(line):
    return any(re.search(pattern, line, re.IGNORECASE) for pattern, _ in _SCAFFOLD)


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
            filled_json, written = fill_template(base_json, parsed)
            stale = unwritten(base_json, filled_json, written)
            if stale:
                raise ParseProblem(
                    ["%s came out identical to the template it was copied from, "
                     "so it would go live about the previous product." % s
                     for s in stale])
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
        # Blocks that ran on and were cut back. The page is right, but the
        # output was not, and that is worth seeing rather than swallowing.
        'trimmed': parsed.get('trimmed', []),
    }

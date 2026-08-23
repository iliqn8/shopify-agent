"""Write generated page copy and colours into a Shopify theme.

The Product Builder tab hands us whatever the model returned. We find the
blocks we recognise, show exactly what would change, and only write when the
caller asks for it.

Three kinds of content, told apart by the field names they carry:

  colours   FIELDS, BY SECTION      -> the product template's colour settings
  copy      SECTION 5 / SECTION 6   -> features grid and comparison table text
  globals   GLOBAL THEME COLOURS    -> config/settings_data.json, whole store

Nothing here prints or exits. Everything returns.
"""
import re
import json
import random
import string
import datetime

import shopify_client as sc
import palette as pal

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
FIELD = re.compile(r"^\s*(?:\d+\s+)?([a-z_]+)\s*=\s*(.+?)\s*$")

TEMPLATE_KEY = "templates/product.gummies.json"
SETTINGS_KEY = "config/settings_data.json"

# Sections that own their copy. Everything else on the page is either a scheme
# reference or has no text of its own.
GRID_TYPE = "custom-protein-features-grid"
TABLE_TYPE = "custom-comparison-table"

GLOBAL_KEYS = ("colors_background_1", "colors_text", "colors_accent_1", "colors_accent_2")
SCHEME_KEYS = ("background", "text", "button", "button_label",
               "secondary_button_label", "shadow")
SCHEME_IDS = ("scheme-1", "scheme-2", "scheme-3", "scheme-4")

# Placeholders a half-finished response leaves behind.
EMPTY = {"", "...", "…", "#......", "-", "n/a", "tbd"}


# ── discovery ──────────────────────────────────────────────────────────────

def list_product_templates():
    """Product templates in the live theme, newest-looking name last."""
    theme = sc.get_active_theme()
    keys = [f["key"] for f in sc.list_theme_files(theme["id"])]
    out = []
    for k in sorted(keys):
        if not (k.startswith("templates/product") and k.endswith(".json")):
            continue
        try:
            tpl = json.loads(sc.get_theme_file(theme["id"], k)["value"])
        except Exception:
            continue
        types = [tpl["sections"][s].get("type")
                 for s in tpl.get("order", []) if s in tpl.get("sections", {})]
        out.append({
            "key": k,
            "label": k.replace("templates/", "").replace(".json", ""),
            "sections": len(types),
            "has_grid": types.count(GRID_TYPE),
            "has_table": types.count(TABLE_TYPE),
        })
    return out


# ── parsing the model's answer ─────────────────────────────────────────────

def _clean(value):
    v = (value or "").strip()
    return "" if v.lower() in EMPTY else v


def _slice(lines, start_pats, stop_pats):
    start = stop = None
    for i, line in enumerate(lines):
        u = line.strip().upper()
        if start is None:
            if any(re.search(p, u) for p in start_pats):
                start = i
            continue
        if any(re.search(p, u) for p in stop_pats):
            stop = i
            break
    if start is None:
        return []
    return lines[start:stop if stop is not None else len(lines)]


def parse_colors(text):
    """Colour fields, read only inside the FIELDS block so the palette summary
    at the top is never mistaken for assignments."""
    out, section, inside = {}, None, False
    for line in text.splitlines():
        u = line.strip().upper()
        if u.startswith("FIELDS, BY SECTION"):
            inside, section = True, None
            continue
        if u.startswith(("SCHEME FIELDS", "DO NOT TOUCH", "GLOBAL THEME", "PALETTE",
                         "CHECKS", "SECTION 5", "SECTION 6", "SECTION 7",
                         "COMPARISON TABLE", "FEATURES GRID", "SELF-CHECK",
                         "CRITICAL RULES", "WORD COUNT")):
            inside, section = False, None
            continue
        if not inside:
            continue
        m = re.match(r"^\s{0,6}(\d{2})\s+([a-z0-9-]+)\s*$", line)
        if m:
            section = int(m.group(1))
            continue
        # a trailing "(rating_stars block)" note is part of the required format
        m = re.match(r"^\s+([a-z_]+)\s*=\s*(#[0-9A-Fa-f]{6})\s*(?:\([^)]*\))?\s*$", line)
        if m and section:
            out[(section, m.group(1))] = m.group(2)
    return out


def parse_grid(text):
    lines = [l.rstrip() for l in text.splitlines()]
    block = _slice(lines,
                   [r"^SECTION 5\b", r"\bFEATURES GRID\b"],
                   [r"^SECTION 6\b", r"^SECTION 7\b", r"\bCOMPARISON TABLE\b",
                    r"^SELF-CHECK", r"^CRITICAL RULES"])
    if not block:
        return None
    heading, feats, cur, column = "", [], {}, "left"
    for line in block:
        u = line.strip().upper()
        if u.startswith("LEFT COLUMN"):
            column = "left"
            continue
        if u.startswith("RIGHT COLUMN"):
            column = "right"
            continue
        m = FIELD.match(line)
        if not m:
            continue
        key, val = m.group(1), _clean(m.group(2))
        if not val:
            continue
        if key == "heading" and not heading:
            heading = val
            continue
        if key not in ("default_icon", "title", "text"):
            continue
        if key in cur:                      # the next benefit has started
            feats.append(cur)
            cur = {}
        cur[key] = val
        cur["column"] = column
    if cur.get("title") or cur.get("text"):
        feats.append(cur)
    if not heading and not feats:
        return None
    return {"heading": heading, "features": feats}


def parse_table(text):
    lines = [l.rstrip() for l in text.splitlines()]
    block = _slice(lines,
                   [r"^SECTION 6\b", r"\bCOMPARISON TABLE\b"],
                   [r"^SECTION 7\b", r"\bPRODUCT DESCRIPTION\b",
                    r"^SELF-CHECK", r"^CRITICAL RULES"])
    if not block:
        return None
    head, rows = {}, []
    for line in block:
        m = re.match(r"^\s*\d+\s+feature\s*=\s*(.+?)\s*$", line)
        if m and _clean(m.group(1)):
            rows.append(_clean(m.group(1)))
            continue
        m = FIELD.match(line)
        if m and m.group(1) in ("heading", "subheading", "featured_title", "other_title"):
            val = _clean(m.group(2))
            if val and m.group(1) not in head:
                head[m.group(1)] = val
    if not head and not rows:
        return None
    return {"head": head, "rows": rows}


STOP_PATS = [r"^SECTION \d", r"^SELF-CHECK", r"^CRITICAL RULES",
             r"\bFIELDS, BY SECTION\b", r"^SECTION SCHEMES\b", r"^SOURCE SWITCHES\b"]


def _parse_schemes(lines):
    """COLOUR SCHEMES: a scheme id on its own line, then its colours indented.

    Also accepts the older one-per-line "scheme-1.background = #..." shape, and
    a whole scheme written on a single line, so an older prompt still lands.
    """
    block = _slice(lines, [r"^COLOUR SCHEMES\b", r"^COLOR SCHEMES\b"],
                   STOP_PATS + [r"^GLOBAL (THEME )?COLOURS?\b"])
    schemes, current = {}, None
    hunt = re.compile(r"([a-z][a-z _]*?)\s*=\s*(#[0-9A-Fa-f]{6})")

    def put(sid, key, val):
        key = key.strip().replace(" ", "_")
        if key in SCHEME_KEYS:
            schemes.setdefault(sid, {})[key] = val

    for line in block + lines:
        m = re.match(r"^\s*(scheme-\d)\.([a-z_ ]+?)\s*=\s*(#[0-9A-Fa-f]{6})", line)
        if m and m.group(1) in SCHEME_IDS:
            put(m.group(1), m.group(2), m.group(3))
            continue
        if line not in block:
            continue
        m = re.match(r"^\s*(scheme-\d)\b(.*)$", line)
        if m and m.group(1) in SCHEME_IDS:
            current = m.group(1)
            for key, val in hunt.findall(m.group(2)):
                put(current, key, val)
            continue
        if current:
            for key, val in hunt.findall(line):
                put(current, key, val)
    return schemes


def parse_globals(text):
    """The global colours block, plus whatever colour schemes were written."""
    lines = [l.rstrip() for l in text.splitlines()]
    block = _slice(lines, [r"^GLOBAL (THEME )?COLOURS?\b"],
                   STOP_PATS + [r"^COLOUR SCHEMES\b", r"^COLOR SCHEMES\b"])
    legacy = {}
    for line in block:
        m = re.match(r"^\s*(colors_[a-z_0-9]+)\s*=\s*(#[0-9A-Fa-f]{6})", line)
        if m and m.group(1) in GLOBAL_KEYS:
            legacy[m.group(1)] = m.group(2)
    schemes = _parse_schemes(lines)
    if not legacy and not schemes:
        return None
    return {"legacy": legacy, "schemes": schemes}


# ── building the change list ───────────────────────────────────────────────

def _new_block_id(prefix, taken):
    while True:
        bid = prefix + "_" + "".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(6))
        if bid not in taken:
            return bid


class _Diff:
    def __init__(self):
        self.rows = []

    def set(self, store, key, value, where, file_label):
        if not isinstance(value, str):
            return
        old = store.get(key)
        if str(old) == value:
            return
        # #fbf4ec and #FBF4EC are the same colour; do not report a change
        if HEX.match(value) and isinstance(old, str) and HEX.match(old.strip()) \
                and old.strip().lower() == value.lower():
            return
        self.rows.append({"file": file_label, "where": where, "field": key,
                          "old": "" if old is None else str(old), "new": value})
        store[key] = value


def _fit_blocks(section, count, block_type, diff, where, file_label):
    blocks = section.setdefault("blocks", {})
    order = section.setdefault("block_order", list(blocks.keys()))
    ids = [b for b in order if blocks.get(b, {}).get("type") == block_type]
    others = [b for b in order if b not in ids]
    while len(ids) < count:
        bid = _new_block_id(block_type, set(blocks))
        blocks[bid] = {"type": block_type, "settings": {}}
        ids.append(bid)
        diff.rows.append({"file": file_label, "where": where, "field": "block added",
                          "old": "", "new": bid})
    for extra in ids[count:]:
        diff.rows.append({"file": file_label, "where": where, "field": "block removed",
                          "old": extra, "new": ""})
        blocks.pop(extra, None)
    ids = ids[:count]
    section["block_order"] = others + ids if others else ids
    return ids


def _check_structure(tpl):
    bad = []
    sections, order = tpl.get("sections", {}), tpl.get("order", [])
    for sid, sec in sections.items():
        if not isinstance(sec, dict):
            continue
        blocks = sec.get("blocks") or {}
        bo = sec.get("block_order") or []
        if set(bo) - set(blocks):
            bad.append("%s: block_order names a block that does not exist" % sid)
        if set(blocks) - set(bo):
            bad.append("%s: a block is missing from block_order" % sid)
    if set(order) - set(sections):
        bad.append("order names a section that does not exist")
    return bad


def build(text, template_key=TEMPLATE_KEY, do_colors=True, do_copy=True,
          do_globals=False, palette_choice=None):
    """Work out every change. Returns the diff, the problems, and the new files.

    Nothing is uploaded. `apply` takes this same result and writes it.
    """
    # A response either names the twelve roles and lets the table in palette.py
    # decide the rest, or spells every field out the old way. Prefer the roles.
    found_palettes = pal.parse_palettes(text)
    # A palette short of a role is an older response that also carries the
    # fields spelled out, so it is left to the path that has always read it.
    palettes = {k: v for k, v in found_palettes.items() if not pal.missing_roles(v)}
    short_of = {k: pal.missing_roles(v) for k, v in found_palettes.items()
                if pal.missing_roles(v)}
    chosen_key, chosen = None, None
    if palettes:
        key = (palette_choice or "").upper()
        if key in palettes:
            chosen_key, chosen = key, palettes[key]
        elif len(palettes) == 1:
            chosen_key, chosen = list(palettes.items())[0]

    colors = parse_colors(text) if do_colors else {}
    grid = parse_grid(text) if do_copy else None
    table = parse_table(text) if do_copy else None
    globs = parse_globals(text) if do_globals else None

    theme = sc.get_active_theme()
    raw_tpl = sc.get_theme_file(theme["id"], template_key)["value"]
    tpl = json.loads(raw_tpl)
    sections, order = tpl["sections"], tpl["order"]
    tpl_label = template_key.replace("templates/", "")

    diff = _Diff()
    problems, notes = [], []

    if palettes and not chosen:
        problems.append("%d palettes were generated. Pick one before applying."
                        % len(palettes))
    if short_of and not palettes and not parse_colors(text):
        for key, roles in sorted(short_of.items()):
            problems.append("Palette %s has no %s, and there are no spelled-out "
                            "fields to fall back on."
                            % (key or "?", ", ".join(roles)))

    # ── colours from a palette ─────────────────────────────────────────
    if chosen and do_colors and not problems:
        where_pal = "palette " + (chosen_key or "")
        for (pos, field, block), (value, stype) in sorted(pal.expand(chosen).items()):
            if pos > len(order):
                continue
            sec = sections.get(order[pos - 1], {})
            if sec.get("type") != stype:
                notes.append("%02d is %s here, not %s, so %s was left alone."
                             % (pos, sec.get("type", "?"), stype, field))
                continue
            if block:
                stores = [b.setdefault("settings", {})
                          for b in (sec.get("blocks") or {}).values()
                          if b.get("type") == block]
            else:
                stores = [sec.setdefault("settings", {})]
            for store in stores:
                diff.set(store, field, value, "%02d %s" % (pos, sec.get("type", "?")),
                         tpl_label)
        colors = {}

    # ── colours, spelled out the old way ───────────────────────────────
    seen = set()
    if colors:
        for i, sid in enumerate(order, 1):
            sec = sections.get(sid, {})
            where = "%02d %s" % (i, sec.get("type", "?"))

            def touch(store):
                for key in list((store or {}).keys()):
                    if (i, key) in colors:
                        seen.add((i, key))
                        diff.set(store, key, colors[(i, key)], where, tpl_label)

            touch(sec.get("settings"))
            for block in (sec.get("blocks") or {}).values():
                touch(block.get("settings"))
        missing = sorted(set(colors) - seen)
        if missing:
            notes.append("%d colour field%s in the response do not exist in %s: %s"
                         % (len(missing), "" if len(missing) == 1 else "s", tpl_label,
                            ", ".join("%02d/%s" % m for m in missing[:8])
                            + ("…" if len(missing) > 8 else "")))

    # ── features grid ──────────────────────────────────────────────────
    grid_sections = [i for i, s in enumerate(order, 1)
                     if sections.get(s, {}).get("type") == GRID_TYPE]
    if grid:
        feats = ([f for f in grid["features"] if f["column"] == "left"]
                 + [f for f in grid["features"] if f["column"] == "right"])
        if len(feats) != 6:
            problems.append("Features grid: %d benefits, expected 6." % len(feats))
        for n, f in enumerate(feats, 1):
            for key in ("title", "text"):
                if not f.get(key):
                    problems.append("Features grid: benefit %d has no %s." % (n, key))
        left = sum(1 for f in feats if f["column"] == "left")
        if feats and left != 3:
            problems.append("Features grid: %d benefits on the left, expected 3." % left)
        icons = [f.get("default_icon") for f in feats if f.get("default_icon")]
        dupes = sorted({i for i in icons if icons.count(i) > 1})
        if dupes:
            problems.append("Features grid: icon used more than once — %s." % ", ".join(dupes))
        if not grid_sections:
            problems.append("This template has no %s section, so the features grid "
                            "copy has nowhere to go." % GRID_TYPE)
        for i in grid_sections:
            sec = sections[order[i - 1]]
            where = "%02d features grid" % i
            if grid["heading"]:
                diff.set(sec.setdefault("settings", {}), "heading",
                         grid["heading"], where, tpl_label)
            ids = _fit_blocks(sec, len(feats), "feature", diff, where, tpl_label)
            for bid, f in zip(ids, feats):
                st = sec["blocks"][bid].setdefault("settings", {})
                diff.set(st, "column", f["column"], where, tpl_label)
                for key in ("title", "text", "default_icon"):
                    if f.get(key):
                        diff.set(st, key, f[key], where, tpl_label)
        if len(grid_sections) > 1:
            notes.append("This template has %d copies of the features grid; "
                         "all of them get the same six benefits." % len(grid_sections))

    # ── comparison table ───────────────────────────────────────────────
    table_sections = [i for i, s in enumerate(order, 1)
                      if sections.get(s, {}).get("type") == TABLE_TYPE]
    if table:
        if len(table["rows"]) != 6:
            problems.append("Comparison table: %d rows, expected 6." % len(table["rows"]))
        for key in ("heading", "featured_title", "other_title"):
            if not table["head"].get(key):
                problems.append("Comparison table: %s is missing." % key)
        if not table_sections:
            problems.append("This template has no %s section, so the comparison "
                            "table copy has nowhere to go." % TABLE_TYPE)
        for i in table_sections:
            sec = sections[order[i - 1]]
            where = "%02d comparison table" % i
            st = sec.setdefault("settings", {})
            for key, val in table["head"].items():
                diff.set(st, key, val, where, tpl_label)
            ids = _fit_blocks(sec, len(table["rows"]), "feature", diff, where, tpl_label)
            for bid, row in zip(ids, table["rows"]):
                diff.set(sec["blocks"][bid].setdefault("settings", {}),
                         "feature", row, where, tpl_label)

    # ── icons that the theme does not have ─────────────────────────────
    if grid:
        try:
            have = {f["key"].split("/")[-1] for f in sc.list_theme_files(theme["id"])
                    if "pfg-icon" in f["key"]}
            unknown = sorted({f.get("default_icon") for f in grid["features"]
                              if f.get("default_icon")} - have)
            if unknown:
                problems.append("Icons not in the theme: %s. Upload them under "
                                "Assets first, or pick from the ones that exist."
                                % ", ".join(unknown))
        except Exception:
            pass

    bad = _check_structure(tpl)
    files = {template_key: json.dumps(tpl, indent=2, ensure_ascii=False)}
    backups = {template_key: raw_tpl}

    # ── global colours ─────────────────────────────────────────────────
    if chosen and do_globals and not problems:
        globs = {"legacy": pal.globals_for(chosen), "schemes": pal.schemes_for(chosen)}
    if globs:
        raw_set = sc.get_theme_file(theme["id"], SETTINGS_KEY)["value"]
        settings = json.loads(raw_set)
        cur = settings["current"]
        for key, val in globs["legacy"].items():
            diff.set(cur, key, val, "theme colours", "settings_data.json")
        # Since the sections moved to colour schemes, these are what actually
        # repaint the page. Writing only scheme-1 leaves the page unchanged.
        for sid in SCHEME_IDS:
            wanted = globs["schemes"].get(sid)
            if not wanted:
                continue
            store = (cur.get("color_schemes", {}).get(sid, {}) or {}).get("settings")
            if store is None:
                problems.append(sid + " is not in this theme, so it cannot be updated.")
                continue
            for key, val in wanted.items():
                diff.set(store, key, val, sid, "settings_data.json")
        files[SETTINGS_KEY] = json.dumps(settings, indent=2, ensure_ascii=False)
        backups[SETTINGS_KEY] = raw_set

    if bad:
        problems.extend(bad)

    found = {
        "palettes": len(palettes),
        "colors": len(colors),
        "grid": len(grid["features"]) if grid else 0,
        "table": len(table["rows"]) if table else 0,
        "globals": (len(globs["legacy"])
                    + sum(len(v) for v in globs["schemes"].values())) if globs else 0,
    }
    if do_colors and not colors:
        if not palettes:
            notes.append("No colour block found. Expected a PALETTE block naming "
                         "the twelve roles, or “FIELDS, BY SECTION” "
                         "followed by the fields.")
    # A section this template has, that the response says nothing about, keeps
    # whatever the template was copied from. That is how a page about vein
    # patches ended up advertising magnesium gummies.
    if do_copy and grid_sections and not grid:
        problems.append(
            "This template has a features grid, but the response has no grid copy. "
            "Applying now would leave the previous product's six benefits on the page.")
    if do_copy and table_sections and not table:
        problems.append(
            "This template has a comparison table, but the response has no table copy. "
            "Applying now would leave the previous product's rows on the page.")
    if do_copy and not grid and not table:
        notes.append("No copy block found. Expected “SECTION 5” or "
                     "“SECTION 6” with field names.")
    if do_globals and not globs and not palettes:
        notes.append("No “GLOBAL COLOURS” block found, and no palette to "
                     "work them out from.")

    return {
        "theme": {"name": theme["name"], "id": theme["id"]},
        "template": template_key,
        # Every palette in the response, so one can be picked and previewed.
        "palettes": palettes,
        "palette_choice": chosen_key,
        "found": found,
        "changes": diff.rows,
        "problems": problems,
        "notes": notes,
        "files": files,
        "backups": backups,
        "safe_to_apply": not problems and bool(diff.rows),
    }


def apply(result):
    """Write the files from a `build` result. Refuses if it flagged problems."""
    if result.get("problems"):
        return {"ok": False, "error": "There are problems to fix first.",
                "problems": result["problems"]}
    if not result.get("changes"):
        return {"ok": False, "error": "Nothing to write."}

    theme_id = result["theme"]["id"]
    written = []
    for key, blob in result["files"].items():
        # only write a file that actually changed
        if not any(c["file"].split("/")[-1] in key for c in result["changes"]):
            continue
        sc.update_theme_file(theme_id, key, blob)
        written.append(key)
    return {"ok": True, "written": written,
            "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


# ── what the prompt has to produce, kept next to the parser that reads it ──

FORMAT_HELP = [
    {
        "id": "colors",
        "title": "Colours for the page",
        "why": "Fills every colour setting in the product template.",
        "needs": "A line reading FIELDS, BY SECTION, then the section number and "
                 "type, then one field per line. The section number is its position "
                 "in the template, counting from the top.",
        "example": "FIELDS, BY SECTION\n"
                   "\n"
                   "  05 custom-protein-features-grid\n"
                   "       background_color                       = #FBF4EC\n"
                   "       heading_color                          = #123F3D\n"
                   "       text_color                             = #5C625D\n"
                   "\n"
                   "  08 custom-comparison-table\n"
                   "       background_color                       = #FBF4EC\n"
                   "       yes_color                              = #123F3D\n"
                   "       no_color                               = #A94B55",
    },
    {
        "id": "grid",
        "title": "Features grid copy",
        "why": "The six benefits around the product photo.",
        "needs": "A heading, then LEFT COLUMN and RIGHT COLUMN, three benefits under "
                 "each. Every benefit needs a title and a text; the icon is optional "
                 "but must be one the theme already has.",
        "example": "SECTION 5 — FEATURES GRID COPY\n"
                   "\n"
                   "heading = A Calmer Way to End Your Day\n"
                   "\n"
                   "LEFT COLUMN\n"
                   "  1  default_icon = pfg-icon-hydration.svg\n"
                   "     title        = Magnesium and Glycine\n"
                   "     text         = Provides 200mg to help support natural relaxation.\n"
                   "\n"
                   "RIGHT COLUMN\n"
                   "  4  default_icon = pfg-icon-mind.svg\n"
                   "     title        = Quiet the Racing Mind\n"
                   "     text         = Helps settle evening thoughts so you drift off.",
    },
    {
        "id": "table",
        "title": "Comparison table copy",
        "why": "The us-versus-them table and its six rows.",
        "needs": "Four header fields, then ROWS with six numbered lines. Field names "
                 "exactly as shown.",
        "example": "SECTION 6 — COMPARISON TABLE COPY\n"
                   "\n"
                   "heading        = Driftwell Vs Regular Sleep Gummies\n"
                   "subheading     = Most sleep gummies lean on melatonin.\n"
                   "featured_title = Driftwell Chews\n"
                   "other_title    = Regular Sleep Gummies\n"
                   "\n"
                   "ROWS\n"
                   "  1  feature = No 3am wake-up\n"
                   "  2  feature = Works without melatonin",
    },
    {
        "id": "globals",
        "title": "Theme colours",
        "why": "The four store colours and scheme-1. These reach the header, the "
               "cart, collections and every other page — not just this product.",
        "needs": "A block headed GLOBAL THEME COLOURS. Only these keys are read.",
        "example": "GLOBAL THEME COLOURS\n"
                   "  colors_background_1                   = #FBF4EC\n"
                   "  colors_text                           = #123F3D\n"
                   "  colors_accent_1                       = #123F3D\n"
                   "  colors_accent_2                       = #DCE9E5\n"
                   "  scheme-1.background                   = #FBF4EC\n"
                   "  scheme-1.text                         = #123F3D",
    },
]

FORMAT_RULES = [
    "Field names are the theme's own. Do not rename or correct them — including "
    "custom_contaner_colors_background, which the theme really does misspell.",
    "One field per line, in the shape name = value. A trailing note in brackets "
    "is fine: star_color = #C6A15B (rating_stars block).",
    "Colours must be full six-digit hex, like #123F3D. Not names, not rgb().",
    "Anything the parser does not recognise is ignored, not guessed at. If a "
    "block is missing, Preview says so instead of writing something wrong.",
]

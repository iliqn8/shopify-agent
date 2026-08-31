# -*- coding: utf-8 -*-
"""A palette of roles, and everywhere on the product page each role goes.

Section 4 of the prompt used to spell out forty hex lines, four colour schemes
and a scheme per section. All of it follows from the twelve roles by a table
that never changes, so the table lives here instead. The response only has to
name the roles, which is what makes three palettes to choose from cheap: the
model writes twelve lines three times rather than a hundred and forty.
"""
import re

ROLES = ("BG", "SURFACE", "INK", "MUTED", "ACCENT", "ACCENT_SOFT",
         "ON_ACCENT", "HAIRLINE", "CONTRAST", "TICK", "NEGATIVE", "STAR")

# role -> (position in the template, the type that position must be, field,
#          block type or None for the section's own settings)
#
# Positions and types are checked before anything is written, so a template
# laid out differently is skipped with a note rather than painted wrongly.
ROLE_FIELDS = {
    "BG": [
        (6,  "benefit-icons-image",              "bg_color", None),
        (8,  "custom-comparison-table",          "highlight_background_color", None),
        (10, "custom-happy-customers-carousel",  "card_background_color", None),
        (15, "wd-section-divider",               "divider_color_top", None),
        (15, "wd-section-divider",               "divider_color_bottom", None),
        # only reach the page on a Custom scheme, filled so the page still
        # holds together if one is ever switched to Custom
        (9,  "rich-text",                        "custom_colors_background", None),
        (11, "ds-testimonials",                  "custom_colors_background", None),
        (12, "collapsible-content",              "custom_colors_background", None),
        (14, "ds-icon-bar",                      "custom_colors_background", None),
    ],
    "SURFACE": [
        (5,  "custom-protein-features-grid",     "icon_background_color", None),
        (13, "custom-why-shop-with-us",          "card_bg_color", None),
        (16, "custom-protein-features-grid",     "icon_background_color", None),
        (12, "collapsible-content",              "custom_contaner_colors_background", None),
    ],
    "INK": [
        (3,  "custom-ugc-video-carousel",        "icon_bg_color", None),
        (3,  "custom-ugc-video-carousel",        "arrow_color", None),
        (5,  "custom-protein-features-grid",     "heading_color", None),
        (5,  "custom-protein-features-grid",     "title_color", None),
        (6,  "benefit-icons-image",              "headline_color", None),
        (10, "custom-happy-customers-carousel",  "verify_badge_color", None),
        (13, "custom-why-shop-with-us",          "heading_color", None),
        (13, "custom-why-shop-with-us",          "title_color", None),
        (16, "custom-protein-features-grid",     "heading_color", None),
        (16, "custom-protein-features-grid",     "title_color", None),
        (9,  "rich-text",                        "custom_colors_text", None),
        (11, "ds-testimonials",                  "custom_colors_text", None),
        (12, "collapsible-content",              "custom_colors_text", None),
        (12, "collapsible-content",              "custom_contaner_colors_text", None),
        (14, "ds-icon-bar",                      "custom_colors_text", None),
    ],
    "MUTED": [
        (5,  "custom-protein-features-grid",     "text_color", None),
        (6,  "benefit-icons-image",              "subhead_color", None),
        (13, "custom-why-shop-with-us",          "subheading_color", None),
        (13, "custom-why-shop-with-us",          "description_color", None),
        (16, "custom-protein-features-grid",     "text_color", None),
    ],
    "ACCENT": [
        (6,  "benefit-icons-image",              "icon_color", None),
    ],
    "ACCENT_SOFT": [
        (5,  "custom-protein-features-grid",     "glow_color", None),
        (13, "custom-why-shop-with-us",          "badge_bg_color", None),
        (13, "custom-why-shop-with-us",          "icon_bg_color", None),
        (16, "custom-protein-features-grid",     "glow_color", None),
    ],
    "ON_ACCENT": [
        (3,  "custom-ugc-video-carousel",        "icon_color", None),
        (9,  "rich-text",                        "custom_colors_solid_button_text", None),
    ],
    "HAIRLINE": [
        (2,  "main-product",                     "bg_star_color", "rating_stars"),
        (3,  "custom-ugc-video-carousel",        "dot_color", None),
        (5,  "custom-protein-features-grid",     "icon_shadow_color", None),
        (8,  "custom-comparison-table",          "row_line_color", None),
        (16, "custom-protein-features-grid",     "icon_shadow_color", None),
    ],
    # the buy button and the nine others follow scheme-4; the sticky bar has no
    # such switch and has to be told the colour by hand
    "CONTRAST": [
        (2,  "main-product",                     "custom_btn_color", "sticky_atc"),
    ],
    "TICK": [
        (8,  "custom-comparison-table",          "yes_color", None),
    ],
    "NEGATIVE": [
        (8,  "custom-comparison-table",          "no_color", None),
    ],
    "STAR": [
        (2,  "main-product",                     "star_color", "rating_stars"),
        (2,  "main-product",                     "star_color", "reviews"),
        (10, "custom-happy-customers-carousel",  "star_color", None),
        (11, "ds-testimonials",                  "stars_color", None),
    ],
}

# Settings that are not colours but decide whether one is used, and the one
# source switch that is deliberately off Scheme 4.
FIXED = [
    (2,  "main-product",            "enable_custom_btn_color", "sticky_atc", True),
    (8,  "custom-comparison-table", "yes_color_source", None, "custom"),
]

# Ground, band, ground, band, with one CTA-coloured band for the guarantee.
SECTION_SCHEMES = {
    1:  ("related-products",                {"color_scheme": "scheme-1"}),
    2:  ("main-product",                    {"color_scheme": "scheme-1"}),
    3:  ("custom-ugc-video-carousel",       {"color_scheme": "scheme-2"}),
    4:  ("image-with-text",                 {"section_color_scheme": "scheme-1",
                                             "color_scheme": "scheme-1"}),
    5:  ("custom-protein-features-grid",    {"color_scheme": "scheme-2"}),
    7:  ("image-with-text",                 {"section_color_scheme": "scheme-1",
                                             "color_scheme": "scheme-1"}),
    8:  ("custom-comparison-table",         {"color_scheme": "scheme-2"}),
    9:  ("rich-text",                       {"color_scheme": "scheme-4"}),
    10: ("custom-happy-customers-carousel", {"color_scheme": "scheme-2"}),
    12: ("collapsible-content",             {"color_scheme": "scheme-1"}),
    13: ("custom-why-shop-with-us",         {"color_scheme": "scheme-2"}),
    16: ("custom-protein-features-grid",    {"color_scheme": "scheme-1"}),
}


def schemes_for(roles):
    """The four colour schemes a palette implies."""
    bg, sur = roles["BG"], roles["SURFACE"]
    ink, acc = roles["INK"], roles["ACCENT"]
    on, con = roles["ON_ACCENT"], roles["CONTRAST"]
    return {
        "scheme-1": {"background": bg,  "text": ink, "button": acc,
                     "button_label": on, "secondary_button_label": ink, "shadow": ink},
        "scheme-2": {"background": sur, "text": ink, "button": acc,
                     "button_label": on, "secondary_button_label": ink, "shadow": ink},
        "scheme-3": {"background": acc, "text": on, "button": on,
                     "button_label": acc, "secondary_button_label": on, "shadow": on},
        "scheme-4": {"background": con, "text": bg, "button": bg,
                     "button_label": con, "secondary_button_label": bg, "shadow": con},
    }


def globals_for(roles):
    """The four legacy colours the header, cart and badges still resolve through."""
    return {"colors_background_1": roles["BG"], "colors_text": roles["INK"],
            "colors_accent_1": roles["ACCENT"], "colors_accent_2": roles["ACCENT_SOFT"]}


def expand(roles):
    """Every template value a palette decides, keyed (position, field, block)."""
    out = {}
    for role, targets in ROLE_FIELDS.items():
        value = roles.get(role)
        if not value:
            continue
        for pos, stype, field, block in targets:
            out[(pos, field, block)] = (value, stype)
    for pos, stype, field, block, value in FIXED:
        out[(pos, field, block)] = (value, stype)
    for pos, (stype, fields) in SECTION_SCHEMES.items():
        for field, value in fields.items():
            out[(pos, field, None)] = (value, stype)
    return out


# ── reading the palettes out of a response ────────────────────────────────

# A-Z, not A-C: "more palettes" letters its new ones D, E, F and onward, and a
# block the parser cannot see is a palette the operator cannot pick.
_HEAD = re.compile(r"^\s*PALETTE(?:\s+([A-Z]))?\b[^\n]*$", re.I | re.M)


def parse_palettes(text):
    """Every PALETTE block, by its letter. A block with no letter keys on ''."""
    out = {}
    heads = list(_HEAD.finditer(text))
    for n, head in enumerate(heads):
        end = heads[n + 1].start() if n + 1 < len(heads) else len(text)
        body = text[head.end():end]
        roles = {}
        for line in body.split(chr(10)):
            m = re.match(r"^\s*([A-Z_]+)\s*=\s*(#[0-9A-Fa-f]{6})\s*$", line)
            if m and m.group(1) in ROLES:
                roles[m.group(1)] = m.group(2)
            elif line.strip() and not re.match(r"^\s*[A-Z_]+\s*=", line):
                # the first line that is not a role ends the block
                if roles:
                    break
        if len(roles) >= 8:
            out[(head.group(1) or "").upper()] = roles
    return out


def missing_roles(roles):
    return [r for r in ROLES if r not in roles]

IMAGE_TYPES = [
    "Hero",
    "Benefits",
    "Features",
    "How To Use",
    "Review",
    "Comparison",
    "Guarantee",
    "Social Proof in Action",
    "Value Props (Description)",
]


def build_brand_dna_prompt(product_title, domain_name, competitor, color_preferences="", additional_notes=""):
    color_note = f"Color preferences from user: {color_preferences}" if color_preferences else "Auto-select colors from the vibe palette below."
    notes_note = f"Additional notes: {additional_notes}" if additional_notes else ""

    return f"""You are a DTC ecommerce brand strategist. Analyze the product image (attached) and the information below to generate complete brand DNA for a 9-image product page set.

Product Title: {product_title}
Domain / Brand Name: {domain_name}
Competitor URL or Description: {competitor}
{color_note}
{notes_note}

VIBE PALETTE DICTIONARY — pick the single best row:

Vibe | Primary BG | Secondary BG | Headline | Accent | Dark Accent | Headline Font | Body Font
Masculine/Edgy/Dark | #1A1A1A | #0F0F0F | #F5F5F0 | #C8102E | #000000 | Bebas Neue | Oswald
Feminine/Soft/Beauty | #FBF1E8 | #F8DCD4 | #3D2817 | #B76E79 | #5C1F2E | Playfair Display | Poppins
Wellness/Clean/Health | #F4EDE0 | #C8D5B9 | #2C2C2C | #4A7856 | #2D4A36 | Inter | Poppins
Tech/Modern/Gadgets | #E8EBED | #0A1F3D | #0A1F3D | #2D7DD2 | #0A1F3D | Space Grotesk | Inter
Premium/Luxury | #0E0E0E | #ECE3D0 | #FAF6EE | #C9A961 | #000000 | Playfair Display | Cormorant Garamond
Bold/Energetic/Fun | #1B2845 | #5BC0EB | #FFFFFF | #FFD60A | #1B2845 | Poppins Black | DM Sans Black
Natural/Organic/Eco | #E8DCC4 | #B5BEA6 | #4A3728 | #4A6741 | #3D2817 | Recoleta | Caprasimo
Cozy/Lifestyle/Warm | #F4C2A1 | #D08770 | #2C2C2C | #CC6B49 | #3A2618 | DM Sans | Recoleta

For ambiguous products: match to the customer's aspirational identity, not the literal product function.

Return ONLY a valid JSON object with exactly these fields (no markdown, no explanation):
{{
  "VIBE_CATEGORY": "e.g. Feminine / Soft / Beauty",
  "PRIMARY_BG_COLOR": "#hex",
  "SECONDARY_BG_COLOR": "#hex",
  "HEADLINE_COLOR": "#hex",
  "ACCENT_COLOR": "#hex",
  "DARK_ACCENT_COLOR": "#hex",
  "HEADLINE_FONT": "font name",
  "BODY_FONT": "font name",
  "DEMOGRAPHIC_ARCHETYPE": "Detailed description of target customer including age range, gender, styling, appearance. End with: Bottom bar profile pics: 3 [description].",
  "PRODUCT_VISUAL_DESCRIPTION": "2-3 sentence visual description of the product's appearance, colors, materials, and distinctive features — for use in image generation prompts.",
  "HEADLINE_HOOK": "3-6 word hero headline",
  "OUTCOME_HEADLINE": "4-7 word outcome headline",
  "QUANTIFIABLE_PROMISE": "measurable claim e.g. 93% report improved well-being",
  "BENEFIT_1": "2-4 word feature",
  "BENEFIT_2": "2-4 word feature",
  "BENEFIT_3": "2-4 word feature",
  "LIFESTYLE_BENEFIT_1": "5-8 word outcome benefit",
  "LIFESTYLE_BENEFIT_2": "5-8 word outcome benefit",
  "LIFESTYLE_BENEFIT_3": "5-8 word outcome benefit",
  "FEATURE_HEADLINE": "e.g. WHAT'S INSIDE? or BUILT TO LAST",
  "FEATURE_1_NAME": "feature name",
  "FEATURE_1_BENEFIT": "3-6 word benefit",
  "FEATURE_2_NAME": "feature name",
  "FEATURE_2_BENEFIT": "3-6 word benefit",
  "FEATURE_3_NAME": "feature name",
  "FEATURE_3_BENEFIT": "3-6 word benefit",
  "FEATURE_4_NAME": "feature name",
  "FEATURE_4_BENEFIT": "3-6 word benefit",
  "STEP_1": "10-14 word first step instruction",
  "STEP_2": "10-14 word second step instruction",
  "STEP_3": "10-14 word third step instruction",
  "CUSTOMER_QUOTE": "8-15 word genuine-sounding customer review",
  "CUSTOMER_NAME": "First name + last initial e.g. Sarah M.",
  "SOCIAL_PROOF_METRIC": "e.g. 10,000+ Satisfied Customers",
  "COMPETITOR_LABEL_1": "generic category label e.g. GENERIC VITAMINS",
  "COMPETITOR_LABEL_2": "generic category label e.g. MAINSTREAM BRANDS",
  "POSITIVE_DIFFERENTIATOR_1": "emoji + 2-3 word positive trait",
  "POSITIVE_DIFFERENTIATOR_2": "emoji + 2-3 word positive trait",
  "POSITIVE_DIFFERENTIATOR_3": "emoji + 2-3 word positive trait",
  "POSITIVE_DIFFERENTIATOR_4": "emoji + 2-3 word positive trait",
  "NEGATIVE_TRAIT_COMPETITORS_HAVE": "emoji + 2-3 word bad trait competitors share",
  "GUARANTEE_HEADLINE": "e.g. 30-DAY MONEY BACK GUARANTEE",
  "BRAND_COMMITMENT_COPY": "2-3 sentence brand promise / guarantee copy",
  "REASSURANCE": "short tagline e.g. Try risk-free today!"
}}"""


def _fill(template, d):
    """Replace {{KEY}} placeholders with values from dict d."""
    result = template
    for key, value in d.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _common_suffix(d):
    headline_font = d.get("HEADLINE_FONT", "Poppins")
    body_font = d.get("BODY_FONT", "Inter")
    product_desc = d.get("PRODUCT_VISUAL_DESCRIPTION", "")
    return f"""

MANDATORY TYPOGRAPHY: Every headline uses {headline_font}. Every body/paragraph text uses {body_font}. These exact fonts must be used consistently — this is image X of 9 and all 9 images share the same typography.
MANDATORY SPELLING: Every word of visible text in this image must be spelled perfectly in English. No typos, no garbled letters, no placeholder text, no nonsense words.
PRODUCT CONSISTENCY: The product appearing in this image must match exactly the reference product image provided: {product_desc} — same packaging shape, same colors, same logo/branding, same visual identity as the uploaded photo."""


def get_image_prompt(index, brand_dna):
    fns = [
        _prompt_hero,
        _prompt_benefits,
        _prompt_features,
        _prompt_how_to_use,
        _prompt_review,
        _prompt_comparison,
        _prompt_guarantee,
        _prompt_social_proof_action,
        _prompt_value_props,
    ]
    return fns[index](brand_dna) + _common_suffix(brand_dna)


def _prompt_hero(d):
    return f"""Create a square 1080x1080 product hero image for a DTC ecommerce product page. Image 1 of a 9-image cohesive brand set. Premium, conversion-focused, zero clutter.

BACKGROUND: Clean {d['PRIMARY_BG_COLOR']} background with subtle texture, soft directional studio lighting from above-right. Subtle vignette darkening corners 12-15%. Lower third slightly darker tone. Premium and intentional, not muddy.

HEADLINE (top third, left-aligned, ~50% of canvas width): Bold uppercase sans-serif in {d['HEADLINE_COLOR']}, heavy weight, tightly kerned, two lines. Reads: "{d['HEADLINE_HOOK']}"

PRODUCT (center to lower-center, 55-65% of canvas height): {d['PRODUCT_VISUAL_DESCRIPTION']} sits on a subtle surface plane with a soft shadow underneath. No thick pedestal or platform. Product is the visual hero.

BENEFIT CARDS (right side, ~38% of canvas width): Three horizontal white rounded-rectangle pill cards stacked vertically. Each card: solid circular icon on left filled {d['ACCENT_COLOR']} with white checkmark, bold black sans-serif benefit text.
- Card 1: "{d['BENEFIT_1']}"
- Card 2: "{d['BENEFIT_2']}"
- Card 3: "{d['BENEFIT_3']}"

LIGHTING: Bright enough to see product textures clearly. Soft even premium product photography lighting. ASPECT RATIO: 1:1 square."""


def _prompt_benefits(d):
    return f"""Create a square 1080x1080 lifestyle benefits image for a DTC ecommerce product page. Image 2 of a 9-image cohesive brand set. Visually distinct from image 1, outcome-forward.

BACKGROUND: Solid {d['SECONDARY_BG_COLOR']} background, noticeably different tone from hero while staying on-brand.

LEFT 60% OF CANVAS:
- Bold uppercase headline in {d['HEADLINE_COLOR']}: "{d['OUTCOME_HEADLINE']}"
- Subhead in {d['ACCENT_COLOR']}: "{d['QUANTIFIABLE_PROMISE']}"
- Three white rounded pill cards stacked:
  - Card 1: "{d['LIFESTYLE_BENEFIT_1']}" with relevant emoji
  - Card 2: "{d['LIFESTYLE_BENEFIT_2']}" with relevant emoji
  - Card 3: "{d['LIFESTYLE_BENEFIT_3']}" with relevant emoji

RIGHT 40% OF CANVAS: A real-world authentic photo of {d['PRODUCT_VISUAL_DESCRIPTION']} being used in its primary use case. Product or its application is the focal point. Real but elevated, not stock-photo posed.

LIGHTING: Bright, lifestyle, slightly cinematic on the right side. ASPECT RATIO: 1:1 square."""


def _prompt_features(d):
    return f"""Create a square 1080x1080 informational features grid image for a DTC ecommerce product page. Image 3 of a 9-image cohesive brand set.

BACKGROUND: {d['SECONDARY_BG_COLOR']} solid.

THREE STRICT HORIZONTAL ZONES — elements NEVER overlap between zones:

ZONE 1 (top 18%): Bold uppercase headline in {d['HEADLINE_COLOR']}: "{d['FEATURE_HEADLINE']}". Centered. Contained entirely within this band.

ZONE 2 (middle 64%): {d['PRODUCT_VISUAL_DESCRIPTION']} centered. Four feature cards in the four corners of this zone — NOT the full canvas corners:
- Top-left: "{d['FEATURE_1_NAME']}" / "{d['FEATURE_1_BENEFIT']}"
- Top-right: "{d['FEATURE_2_NAME']}" / "{d['FEATURE_2_BENEFIT']}"
- Bottom-left: "{d['FEATURE_3_NAME']}" / "{d['FEATURE_3_BENEFIT']}"
- Bottom-right: "{d['FEATURE_4_NAME']}" / "{d['FEATURE_4_BENEFIT']}"
Each card: white rounded rectangle, soft drop shadow, small product-relevant photo or icon, bold black heading, lighter subheading.

ZONE 3 (bottom 18%): Breathing room only. No text or cards.

STYLE: Premium DTC educational layout. ASPECT RATIO: 1:1 square."""


def _prompt_how_to_use(d):
    return f"""Create a square 1080x1080 instructional how-to-use image for a DTC ecommerce product page. Image 4 of a 9-image cohesive brand set.

BACKGROUND: Soft gradient combining {d['PRIMARY_BG_COLOR']} at top fading to a complementary lighter shade at bottom.

TWO STRICT HORIZONTAL ZONES:

ZONE 1 (top 12%): Bold uppercase headline in {d['HEADLINE_COLOR']}: "HOW TO USE". Centered. Fully separate from panels below.

ZONE 2 (bottom 88%): Three vertical panels of equal width side-by-side with rounded corners and thin spacing between them.

Each panel top 60%: lifestyle photo showing the step.
- Panel 1 photo: A hand performing the first action with {d['PRODUCT_VISUAL_DESCRIPTION']} (opening, scooping, unpacking)
- Panel 2 photo: A hand performing the core action (applying, mixing, activating, pressing)
- Panel 3 photo: The final result or enjoyment moment

Each panel bottom 40% (white background):
- Pill badge in {d['DARK_ACCENT_COLOR']}: "Step 1" / "Step 2" / "Step 3" in white bold sans-serif
- Instruction text in black sans-serif, key action bolded
- Step 1: "{d['STEP_1']}"
- Step 2: "{d['STEP_2']}"
- Step 3: "{d['STEP_3']}"

Photos across all three panels: consistent background surface, lighting direction, and brand vibe. ASPECT RATIO: 1:1 square."""


def _prompt_review(d):
    return f"""Create a square 1080x1080 social proof and review image for a DTC ecommerce product page. Image 5 of a 9-image cohesive brand set.

BACKGROUND: Real lifestyle scene fitting the product's typical use environment with {d['PRIMARY_BG_COLOR']} tones in wall color or surface.

LEFT 55% OF CANVAS: White rounded-corner speech-bubble review card with strong drop shadow:
- Small circular {d['DARK_ACCENT_COLOR']} badge at top with white quotation-mark glyph
- Five gold filled stars (warm gold #FFB800)
- Bold black quote: "{d['CUSTOMER_QUOTE']}"
- Below: "{d['CUSTOMER_NAME']} | Verified Review" in smaller gray text

RIGHT 45% OF CANVAS: A real-looking person matching this demographic: {d['DEMOGRAPHIC_ARCHETYPE']} — smiling, looking slightly off-camera with genuine joy, holding or actively using {d['PRODUCT_VISUAL_DESCRIPTION']}. DEMOGRAPHIC MATCHING IS NON-NEGOTIABLE. The person must look exactly like the brand's target customer.

BOTTOM BAR (full width, ~10% height, solid {d['DARK_ACCENT_COLOR']}): Five gold stars, 3 overlapping circular profile photos ALL matching the demographic, bold white text: "{d['SOCIAL_PROOF_METRIC']}"

LIGHTING: Bright natural daylight or warm interior fitting the brand mood. ASPECT RATIO: 1:1 square."""


def _prompt_comparison(d):
    product_title = d.get("product_title", "Our Brand")
    brand_name = product_title.split()[0].upper() if product_title else "BRAND"

    return f"""Create a square 1080x1080 comparison chart image for a DTC ecommerce product page. Image 6 of a 9-image cohesive brand set.

BACKGROUND: Blurred softly-lit texture in {d['PRIMARY_BG_COLOR']} tones hinting at the product's category. Subtle, not competing with the chart.

THREE-COLUMN COMPARISON TABLE centered on canvas:

HEADER ROW:
- Left: Small photo of {d['PRODUCT_VISUAL_DESCRIPTION']} slightly tilted above the brand column
- Center column: Solid {d['DARK_ACCENT_COLOR']} vertical banner running full table height, "{brand_name}" wordmark in white uppercase at top
- Right columns: "{d['COMPETITOR_LABEL_1']}" and "{d['COMPETITOR_LABEL_2']}" in bold uppercase

5 ROWS (white rounded-rectangle cards with drop shadow):
- Row 1: "{d['POSITIVE_DIFFERENTIATOR_1']}" — Brand ✅ — Comp1 ❌ — Comp2 ❌
- Row 2: "{d['POSITIVE_DIFFERENTIATOR_2']}" — Brand ✅ — Comp1 ❌ — Comp2 ❌
- Row 3: "{d['POSITIVE_DIFFERENTIATOR_3']}" — Brand ✅ — Comp1 ❌ — Comp2 ❌
- Row 4: "{d['POSITIVE_DIFFERENTIATOR_4']}" — Brand ✅ — Comp1 ❌ — Comp2 ❌
- Row 5 (THE FLIP): "{d['NEGATIVE_TRAIT_COMPETITORS_HAVE']}" — Brand ❌ — Comp1 ✅ — Comp2 ✅

Check icons: green rounded squares with white checkmark. X icons: red rounded squares with white X.
STYLE: Sharp, scannable, premium DTC infographic. ASPECT RATIO: 1:1 square."""


def _prompt_guarantee(d):
    return f"""Create a square 1080x1080 trust and guarantee closing image for a DTC ecommerce product page. Image 7 of a 9-image cohesive brand set.

BACKGROUND: Solid {d['PRIMARY_BG_COLOR']} background. Warm, confidence-inspiring, welcoming.

LEFT 55% OF CANVAS:
- Circular badge/seal icon in {d['DARK_ACCENT_COLOR']} with scalloped border, containing a ribbon or thumbs-up glyph
- Bold uppercase headline in {d['HEADLINE_COLOR']}: "{d['GUARANTEE_HEADLINE']}"
- 2-3 sentence subhead in lighter weight: "{d['BRAND_COMMITMENT_COPY']}"
- Horizontal pill trust bar: 3 overlapping circular profile photos of people matching: {d['DEMOGRAPHIC_ARCHETYPE']} on left, bold dark text: "{d['SOCIAL_PROOF_METRIC']}" on right

RIGHT 45% OF CANVAS: ONE hero {d['PRODUCT_VISUAL_DESCRIPTION']} positioned prominently, slightly tilted at a dynamic angle. Large, clearly visible, well-lit. Single product only, clean composition.

LIGHTING: Warm directional on the right. Product is well-lit with dimensionality. Left side feels grounded and trustworthy.
STYLE: Premium DTC closing image that makes someone tap Add to Cart. ASPECT RATIO: 1:1 square."""


def _prompt_social_proof_action(d):
    return f"""Create a square 1080x1080 social proof lifestyle collage image for a DTC ecommerce product page. Image 8 of a 9-image cohesive brand set.

BACKGROUND: {d['PRIMARY_BG_COLOR']} background.

LAYOUT: A dynamic collage of 3 authentic lifestyle photos arranged in an engaging grid with thin {d['DARK_ACCENT_COLOR']} borders. Each photo features a different real-looking person matching this demographic: {d['DEMOGRAPHIC_ARCHETYPE']} — naturally using or enjoying {d['PRODUCT_VISUAL_DESCRIPTION']} in real-life settings. Genuine expressions, natural moments, not posed.

TOP OVERLAY BANNER (slim, {d['DARK_ACCENT_COLOR']} background): Bold white uppercase text: "REAL PEOPLE. REAL RESULTS."

BOTTOM BAR (full width, solid {d['DARK_ACCENT_COLOR']}): Five gold stars on left, bold {d['ACCENT_COLOR']} text center: "{d['SOCIAL_PROOF_METRIC']}", right side: 3 overlapping circular profile photos of people matching the demographic.

LIGHTING: Natural lifestyle lighting throughout. Consistent warm tone matching the brand palette. Authentic, approachable.
STYLE: UGC-adjacent, premium but real, on-demographic. ASPECT RATIO: 1:1 square."""


def _prompt_value_props(d):
    return f"""Create a square 1080x1080 value propositions product description image for a DTC ecommerce product page. Image 9 of a 9-image cohesive brand set. Used in the product description section of a Shopify store.

BACKGROUND: {d['SECONDARY_BG_COLOR']} clean professional background.

TOP SECTION (top 20%): Large bold uppercase headline in {d['HEADLINE_COLOR']}: "WHY YOU'LL LOVE IT". Centered.

CENTER SECTION (middle 55%): 2x2 grid of 4 rounded white value proposition cards with soft drop shadows:
- Card 1: Large relevant emoji + bold "{d['BENEFIT_1']}" + 1-line benefit subtext
- Card 2: Large relevant emoji + bold "{d['BENEFIT_2']}" + 1-line benefit subtext
- Card 3: Large relevant emoji + bold "{d['BENEFIT_3']}" + 1-line benefit subtext
- Card 4: Shield or checkmark emoji + bold "Risk-Free" + subtext "{d['GUARANTEE_HEADLINE']}"

BOTTOM SECTION (bottom 25%): {d['PRODUCT_VISUAL_DESCRIPTION']} centered as a clean product shot. Below it: "{d['QUANTIFIABLE_PROMISE']}" in {d['ACCENT_COLOR']} bold text. Below that: "{d['REASSURANCE']}" in smaller {d['HEADLINE_COLOR']} text.

STYLE: Clean, scannable, premium DTC informational image. Information-dense but visually breathable. ASPECT RATIO: 1:1 square."""

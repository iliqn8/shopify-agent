import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

PROMPT_TEMPLATE = """ROLE
You are an 8-figure ecommerce store owner who specializes in branded dropshipping. Your goal is to help me launch a high-converting product page today. You write like a real operator, not like AI. Use AIDA copywriting principles. Lead with benefits and outcomes, not features. Speak directly to the customer.

OUTPUT FORMAT
Output everything as plain typed text only. No HTML. No code blocks. No files. Just type the sections directly in your response, in order.

INPUTS
Product name: [PRODUCT_NAME]
Competitor URL: [COMPETITOR_URL]

My cost of goods:
- Product cost: $[PRODUCT_COST]
- Shipping cost: $[SHIPPING_COST]

IMPORTANT: COGs have already been provided above. Do NOT ask for them. Generate all sections immediately.

SECTION 1 — BRANDED PRODUCT TITLE
Pick ONE branded title only. Do not output three options. Do not output a list. Pick the single best fit and commit to it.

Choose from one of these three angles based on what fits the product best:
Angle A — Branded one-word name: A clean, made-up brand word + the product. 1-2 words max. Easy to pronounce. Unisex. Use this when the product is everyday or not inherently viral.
Angle B — Domain-style name: A made-up store name placed before the product. The name sounds like a real brand someone would Google. Use this when the product benefits from feeling like a legit established brand (wellness, skincare, supplements, beauty).
Angle C — Viral-leaning name: Use the word "Viral" or a trend-aligned descriptor. Only use this when the product is genuinely TikTok-driven.

Output exactly ONE title plus a one-to-two sentence reason it's the right fit. Do not output the other two angles.

SECTION 2 — PRICING + OFFER
Use my COGs (product cost + shipping cost). Calculate and recommend pricing using the strict rules below.

PRICING LADDER (STRICT): All selling prices must end in .95 only. Use only these prices: $24.95, $29.95, $34.95, $39.95, $44.95, $49.95, $54.95, $59.95, $64.95, $69.95, $74.95, $79.95, $84.95, $89.95. Never generate prices outside this ladder.

PROFIT FLOOR (STRICT): Every recommended offer must generate a minimum +$20 profit per order. If a price + offer combination drops below +$20 profit, raise the price to the next ladder step.

OFFER LOGIC:
- COGs $4-$11: Default to BOGO (Buy One Get One Free) with FREE SHIPPING. Calculate profit on 2 units total COGs.
- COGs $12-$15: Default to "Free Shipping on 2+" with $4.95 shipping on 1 unit.
- COGs $16-$20: Default to single unit + FREE SHIPPING + free complimentary gift.
- COGs $21-$25: Default to single unit at $59.95-$74.95 + free shipping.
- COGs above $25: Recommend finding a different product.

OUTPUT FOR THIS SECTION:
- Recommended selling price (from the ladder)
- Recommended offer with one-sentence reasoning
- Total customer pays (including shipping if applicable)
- Profit per order
- Breakeven ROAS (selling price / profit per order)
- Confirm the $20 profit floor is met

SECTION 3 — STORE COLOR PALETTE
Match the product to one of the vibes below based on its category and aspirational customer.

Output exactly 4 values labeled exactly like this:
Background color: #XXXXXX
Text color: #XXXXXX
Accent 1 (buttons): #XXXXXX
Accent 2: #XXXXXX

Then explain in one sentence why this palette fits.

VIBE PALETTES:
Masculine/Edgy/Dark: Background #1A1A1A, Text #F5F5F0, Accent1 #C8102E, Accent2 #2A2A2A
Feminine/Soft/Beauty: Background #F8DCD4, Text #3D2817, Accent1 #B76E79, Accent2 #E8C4BA
Wellness/Clean/Health: Background #F4EDE0, Text #2C2C2C, Accent1 #4A7856, Accent2 #D9C9A8
Tech/Modern/Gadgets: Background #E8EBED, Text #000000, Accent1 #2D7DD2, Accent2 #D0D8E0
Premium/Luxury: Background #0E0E0E, Text #FAF6EE, Accent1 #C9A961, Accent2 #1A1A1A
Bold/Energetic/Fun: Background #FFD60A, Text #000000, Accent1 #FF4081, Accent2 #1B2845
Natural/Organic/Eco: Background #E8DCC4, Text #4A3728, Accent1 #4A6741, Accent2 #C9B99A
Cozy/Lifestyle/Warm: Background #F4C2A1, Text #2C2C2C, Accent1 #CC6B49, Accent2 #E8A882

SECTION 4 — PRODUCT DESCRIPTION
Write the full product page copy following the structure below. Use AIDA principles. Lead with benefits and outcomes. Speak directly to the customer.

LENGTH RULES (STRICT): Paragraphs must be 1 to 2 sentences max. Always split into multiple short paragraphs with a line break between each. Never write a chunky 3+ sentence block.

BOLDING RULES (STRICT): Inside every paragraph in Main Body Sections 1, 2, and 3, bold 1 to 2 specific words or short phrases using **example**. Bold the punchy keyword or benefit, never a full sentence.

OTHER RULES: Never use em dashes. Write like a real person, no AI slop. Use original copy.

DESCRIPTION FORMAT:

Top of Page (above the fold)
3 emoji bullets, each 4-7 words, benefit-driven:

Collapsible Tab — How It Works
3 numbered steps with bolded action verbs. 1 short sentence each.

Collapsible Tab — Reviews
3 five-star reviews, 1-2 sentences each:
"Review text"
— Name, Location

Main Body Section 1
Headline featuring the main benefit.
2 short paragraphs (1-2 sentences each), bold 1-2 keywords per paragraph.

Main Body Section 2
Headline describing how the customer will feel.
4 emoji bullet blocks, each with:
[emoji]
**Bold 2-3 word headline**
Short 4-7 word description sentence.

Main Body Section 3 (Objection Handler)
Headline that names a customer objection or hesitation.
2 short paragraphs (1-2 sentences each), bold 1-2 reassuring words per paragraph.

30-Day Guarantee
1-2 sentences themed around this product's benefit.

FAQ
4 questions with 2-3 sentence answers covering common objections.

SELF-CHECK BEFORE YOU SEND: Verify — ONE title only, paragraphs 1-2 sentences, bold keywords in every Main Body paragraph, Section 2 uses emoji+headline+description format, zero em dashes, price from strict ladder, $20 profit floor met, exactly 4 color values, written like a real operator. If all yes, output.
"""


def build_stream(product_name, competitor_url, product_cost, shipping_cost, images=None):
    """Generator yielding {type: status/done} events."""
    prompt = (PROMPT_TEMPLATE
              .replace("[PRODUCT_NAME]", product_name)
              .replace("[COMPETITOR_URL]", competitor_url or "no competitor URL provided")
              .replace("[PRODUCT_COST]", str(product_cost))
              .replace("[SHIPPING_COST]", str(shipping_cost)))

    yield {"type": "status", "text": "🛍️ Building your product page..."}

    content = [{"type": "text", "text": prompt}]
    if images:
        img_blocks = []
        for img in images[:3]:
            img_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img["media_type"], "data": img["b64"]}
            })
        content = img_blocks + content

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": content}],
            timeout=180.0,
        )
        text = response.content[0].text
        yield {"type": "done", "reply": text}
    except Exception as e:
        yield {"type": "done", "reply": f"Error: {e}"}

import os

SKILLS_DIR = os.path.join(os.path.dirname(__file__), ".claude", "skills")

# keyword (lowercase) -> skill folder name(s)
SKILL_MAP = [
    (["email", "имейл", "newsletter", "мейл", "бюлетин"],           ["emails", "copywriting"]),
    (["cold email", "студен имейл", "outreach"],                     ["cold-email"]),
    (["facebook ads", "google ads", "реклам", " ads "],              ["ads", "ad-creative"]),
    (["social", "социални", "instagram", "tiktok", "пост", "post"],  ["social", "copywriting"]),
    (["seo", "класиране", "search engine"],                          ["seo-audit", "ai-seo"]),
    (["страниц", "product page", "листинг", "listing"],             ["product-marketing", "copywriting"]),
    (["конверс", "conversion", "checkout"],                          ["cro"]),
    (["цен", "price", "pricing", "ценообраз"],                       ["pricing", "offers"]),
    (["промоц", "discount", "отстъпк", "offer", "купон"],            ["offers"]),
    (["launch", "пускане", "стартирам"],                             ["launch", "product-marketing"]),
    (["конкурент", "competitor"],                                     ["competitors", "competitor-profiling"]),
    (["аналит", "analytics", "статистик", "метрик"],                 ["analytics"]),
    (["маркетинг план", "marketing plan", "стратегия"],              ["marketing-plan"]),
    (["маркетинг идеи", "marketing ideas", "идеи за"],               ["marketing-ideas"]),
    (["задърж", "retention", "churn"],                               ["churn-prevention"]),
    (["реферал", "referral", "препоръч", "affiliate"],               ["referrals"]),
    (["блог", "blog", "content strategy", "съдържание"],             ["content-strategy"]),
    (["lead magnet", "лийд"],                                        ["lead-magnets"]),
    (["popup", "изскачащ"],                                          ["popups"]),
    (["onboarding", "онбординг"],                                    ["onboarding"]),
    (["a/b test", "сплит тест", "ab test"],                          ["ab-testing"]),
    (["pr ", "пиар", "медии", "public relations"],                   ["public-relations"]),
    (["видео", "video", "youtube", "reels"],                         ["video"]),
    (["sms", "смс"],                                                 ["sms"]),
    (["копирайт", "copywriting", "текст за", "опис"],                ["copywriting"]),
    (["психолог", "psychology"],                                     ["marketing-psychology"]),
]

MAX_SKILLS = 4


def _load_skill(skill_name):
    path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(6000)  # cap per skill to avoid token explosion


def get_relevant_skills(message):
    """Return combined skill content relevant to the user message. Empty string if none match."""
    if not message or not os.path.exists(SKILLS_DIR):
        return ""

    msg = message.lower()
    matched = []
    seen = set()

    for keywords, skills in SKILL_MAP:
        if any(kw in msg for kw in keywords):
            for s in skills:
                if s not in seen:
                    seen.add(s)
                    matched.append(s)

    if not matched:
        return ""

    parts = []
    for skill_name in matched[:MAX_SKILLS]:
        content = _load_skill(skill_name)
        if content:
            parts.append(f"### [{skill_name}]\n{content}")

    return "\n\n---\n\n".join(parts) if parts else ""

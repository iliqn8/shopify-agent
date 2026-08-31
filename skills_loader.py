"""Marketing skills, exposed as a tool instead of injected into every prompt.

The agent sees a short catalogue of every skill in its system prompt and calls
load_skill() when one is actually relevant. Nothing is truncated on load — the
old loader capped each skill at 6000 characters and never opened references/.
"""

import os
import re

SKILLS_DIR = os.path.join(os.path.dirname(__file__), ".claude", "skills")

# A whole SKILL.md plus a reference file can be long; this only guards against a
# pathological file, it is not the routine trim the old loader did.
MAX_CHARS = 60000

_catalogue_cache = None


def _read_frontmatter(path):
    """Return the frontmatter dict of a SKILL.md. Values stay strings."""
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(8000)
    except OSError:
        return out
    if not head.startswith("---"):
        return out
    end = head.find("\n---", 3)
    if end == -1:
        return out
    for line in head[3:end].split("\n"):
        m = re.match(r'^(\w[\w-]*):\s*(.*)$', line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def _first_sentence(text, limit=200):
    """Trim a long skill description down to something routable."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    if dot > 60:
        return cut[:dot + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > 60 else cut).rstrip(",;") + "..."


def list_skills():
    """Every available skill as {name, description, references}. Cached."""
    global _catalogue_cache
    if _catalogue_cache is not None:
        return _catalogue_cache
    if not os.path.isdir(SKILLS_DIR):
        _catalogue_cache = []
        return _catalogue_cache

    skills = []
    for name in sorted(os.listdir(SKILLS_DIR)):
        skill_md = os.path.join(SKILLS_DIR, name, "SKILL.md")
        if not os.path.exists(skill_md):
            continue
        fm = _read_frontmatter(skill_md)
        skills.append({
            "name": name,
            "description": fm.get("description", ""),
            "references": list_references(name),
        })
    _catalogue_cache = skills
    return skills


def list_references(skill_name):
    """Reference filenames that ship with a skill (empty list if none)."""
    ref_dir = os.path.join(SKILLS_DIR, skill_name, "references")
    if not os.path.isdir(ref_dir):
        return []
    return sorted(f for f in os.listdir(ref_dir)
                  if os.path.isfile(os.path.join(ref_dir, f)))


def catalogue_text():
    """The one-line-per-skill listing that goes into the system prompt."""
    skills = list_skills()
    if not skills:
        return ""
    lines = []
    for s in skills:
        desc = _first_sentence(s["description"]) if s["description"] else "(no description)"
        extra = f" [refs: {', '.join(s['references'])}]" if s["references"] else ""
        lines.append(f"- **{s['name']}** — {desc}{extra}")
    return "\n".join(lines)


def load_skill(skill_name, reference=None):
    """Full text of a skill, or one of its reference files.

    Returns a dict so the tool result carries its own error text instead of
    raising into the agent loop.
    """
    if not skill_name:
        return {"error": "No skill name given."}

    # Keep the name a plain folder name — it arrives from the model.
    safe = os.path.basename(str(skill_name).strip())
    skill_dir = os.path.join(SKILLS_DIR, safe)
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.exists(skill_md):
        names = [s["name"] for s in list_skills()]
        near = [n for n in names if safe in n or n in safe]
        return {"error": f"No skill called '{safe}'.",
                "did_you_mean": near[:5] or names[:10]}

    if reference:
        safe_ref = os.path.basename(str(reference).strip())
        ref_path = os.path.join(skill_dir, "references", safe_ref)
        if not os.path.exists(ref_path):
            return {"error": f"'{safe}' has no reference file '{safe_ref}'.",
                    "available": list_references(safe)}
        with open(ref_path, "r", encoding="utf-8", errors="ignore") as f:
            body = f.read(MAX_CHARS)
        return {"skill": safe, "reference": safe_ref, "content": body}

    with open(skill_md, "r", encoding="utf-8", errors="ignore") as f:
        body = f.read(MAX_CHARS)
    return {"skill": safe, "content": body, "references": list_references(safe)}

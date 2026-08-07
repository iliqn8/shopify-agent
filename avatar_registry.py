"""One registry for every talking-head (UGC) provider.

Providers differ in three ways the rest of the app should not have to know
about: how an actor photo becomes a usable reference, what a voice id looks
like, and whether clip length is controllable. This module normalises all
three so `video_cloner` can treat them identically.

Adding a provider (Arcads is the likely next one) means writing a client module
and one entry in AVATAR_MODELS — no changes to the orchestrator or the UI.
"""

import fal_client
import heygen_client


class AvatarError(Exception):
    pass


AVATAR_MODELS = {
    "heygen": {
        "provider": "heygen",
        "label": "HeyGen — $0.05/s, best quality per dollar",
        "usd_per_second": 0.05,
        # HeyGen decides length from the script; we cannot ask for N seconds.
        "fixed_length": True,
    },
    "ai-avatar": {
        "provider": "fal",
        "fal_key": "ai-avatar",
        "label": "fal AI Avatar — $0.20/s (480p)",
        "usd_per_second": 0.20,
        "fixed_length": False,
    },
    "infinitalk": {
        "provider": "fal",
        "fal_key": "infinitalk",
        "label": "fal InfiniTalk — $0.20/s, longer takes",
        "usd_per_second": 0.20,
        "fixed_length": False,
    },
}

DEFAULT_AVATAR_MODEL = "heygen"


def spec(model_key):
    s = AVATAR_MODELS.get(model_key)
    if not s:
        raise AvatarError(f"Unknown avatar model '{model_key}'")
    return s


def provider_of(model_key):
    return spec(model_key)["provider"]


def is_fixed_length(model_key):
    return spec(model_key)["fixed_length"]


# ── Account status ─────────────────────────────────────────────────────────

def check_provider(provider):
    if provider == "heygen":
        return heygen_client.check_account()
    if provider == "fal":
        return fal_client.check_account()
    return False, f"Unknown provider '{provider}'"


def providers_status():
    """Status of every provider that backs at least one registered model."""
    out = {}
    for p in sorted({m["provider"] for m in AVATAR_MODELS.values()}):
        ok, msg = check_provider(p)
        out[p] = {"ok": ok, "message": msg}
    return out


# ── Voices ─────────────────────────────────────────────────────────────────

def list_voices(model_key):
    """Voice options as [{id, name}] for the given model's provider."""
    if provider_of(model_key) == "heygen":
        try:
            return [{"id": v["id"], "name": v["name"]} for v in heygen_client.list_voices()]
        except Exception:
            return []
    return [{"id": v, "name": v} for v in fal_client.AVATAR_VOICES]


# ── Actor photo ────────────────────────────────────────────────────────────

def upload_actor(model_key, image_bytes, filename="actor.jpg", content_type="image/jpeg"):
    """Turn an uploaded photo into whatever reference this provider needs.

    Returns an opaque string that must be passed back to `generate`.
    """
    if provider_of(model_key) == "heygen":
        return heygen_client.upload_talking_photo(image_bytes, content_type)
    return fal_client.upload_bytes(image_bytes, filename, content_type)


# ── Generation ─────────────────────────────────────────────────────────────

def generate(model_key, actor_ref, script, voice=None, seconds=6,
             scene_prompt=None, aspect_ratio="9:16", on_status=None):
    """Render one talking-head clip. Returns its URL."""
    s = spec(model_key)

    if s["provider"] == "heygen":
        return heygen_client.generate_video(
            actor_ref, script,
            voice_id=voice or None,
            aspect_ratio=aspect_ratio,
            on_status=on_status,
        )

    return fal_client.generate_avatar(
        s["fal_key"], actor_ref, script,
        voice=voice or "Sarah",
        seconds=seconds,
        scene_prompt=scene_prompt,
        on_status=on_status,
    )


def usd_per_second(model_key, resolution="480p"):
    s = spec(model_key)
    price = s["usd_per_second"]
    return price.get(resolution, 0.20) if isinstance(price, dict) else price

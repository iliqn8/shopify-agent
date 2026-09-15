"""CJ Dropshipping API v2 — product search for the Product Hunter tab.

What CJ really does, measured against the live API on 15 September 2026:

- There is NO order or sales count. The closest signal is `listedNum`: how
  many CJ users added the product to their stores. The UI calls it "listings".
- Search runs on the older `/product/list`, not `listV2`. `listV2` matches
  keywords well but returns `createAt: null` for every product, and its
  `timeStart/timeEnd` do not filter the results (same first products with and
  without, totals that go UP when the window narrows). `/product/list` filters
  listings, creation date, warehouse and free shipping on the server, returns
  the creation date, and takes 200 per page.
- Its one weakness: `productNameEn` is an OR match — "massage gun" returns hair
  combs. So every word of the keyword is required here, on the name, after the
  fetch. That is why `pages` exists: a narrow keyword needs more scanned
  products to find enough real matches.
- Free accounts get 1 request per second. The access token lives 180 days and
  is cached on disk (the Railway volume when there is one).
"""

import os
import re
import json
import time
import hashlib
import threading

import requests

BASE = "https://developers.cjdropshipping.com/api2.0/v1"

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(__file__)
_TOKEN_FILE = os.path.join(_DATA_DIR, "cj_token.json")

# Free tier is 1 req/s. A little over a second between calls keeps a scan of
# several pages from tripping the limiter halfway through.
MIN_GAP = 1.1
_lock = threading.Lock()
_last_call = [0.0]
_token = {}

PAGE_SIZE = 200
MAX_PAGES = 5

# /product/list sorts on two fields only. Price order is applied in the
# browser, over what was found.
SORTS = {
    "newest":     {"label": "Newest first",  "orderBy": "createAt",  "sort": "desc"},
    "listed":     {"label": "Most listed",   "orderBy": "listedNum", "sort": "desc"},
    "listed_asc": {"label": "Least listed",  "orderBy": "listedNum", "sort": "asc"},
}
DEFAULT_SORT = "newest"

COUNTRIES = {"": "Any warehouse", "CN": "China", "US": "United States", "GB": "United Kingdom",
             "DE": "Germany", "FR": "France", "IT": "Italy", "ES": "Spain", "PL": "Poland",
             "CZ": "Czechia", "AU": "Australia", "CA": "Canada"}


class CJError(Exception):
    pass


def api_key():
    return (os.getenv("CJ_API_KEY") or "").strip()


def _key_id(key):
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _throttle():
    with _lock:
        wait = MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _load_cached_token(key):
    if _token.get("key_id") == _key_id(key):
        return _token
    try:
        with open(_TOKEN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("key_id") == _key_id(key):
            _token.clear()
            _token.update(data)
            return _token
    except (OSError, ValueError):
        pass
    return None


def _save_token(key, data):
    record = {"key_id": _key_id(key), "access": data["accessToken"],
              "refresh": data.get("refreshToken"), "saved_at": time.time()}
    _token.clear()
    _token.update(record)
    try:
        with open(_TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f)
    except OSError:
        pass   # memory cache still works; only a restart would re-fetch


def _get_token(force=False):
    key = api_key()
    if not key:
        raise CJError("CJ_API_KEY is not set — add it to .env locally and to Railway → Variables.")
    cached = None if force else _load_cached_token(key)
    if cached:
        return cached["access"]
    _throttle()
    r = requests.post(f"{BASE}/authentication/getAccessToken",
                      json={"apiKey": key}, timeout=20)
    body = _json(r)
    if not body.get("result") or not (body.get("data") or {}).get("accessToken"):
        raise CJError(f"CJ rejected the API key: {body.get('message') or r.status_code}")
    _save_token(key, body["data"])
    return _token["access"]


def _json(r):
    try:
        return r.json()
    except ValueError:
        raise CJError(f"CJ returned a non-JSON response ({r.status_code})")


def _looks_like_bad_token(r, body):
    msg = str(body.get("message") or "").lower()
    return r.status_code == 401 or "token" in msg and ("invalid" in msg or "expired" in msg)


def _get(path, params):
    """GET with the cached token, re-authenticating once if CJ says it is stale."""
    for attempt in (1, 2):
        token = _get_token(force=attempt == 2)
        _throttle()
        r = requests.get(f"{BASE}{path}", params=params,
                         headers={"CJ-Access-Token": token}, timeout=30)
        body = _json(r)
        if body.get("result") or body.get("code") == 200:
            return body.get("data")
        if attempt == 1 and _looks_like_bad_token(r, body):
            continue
        raise CJError(f"CJ error {body.get('code')}: {body.get('message') or r.status_code}")


def check_account():
    if not api_key():
        return False, "⚠️ CJ_API_KEY is not set"
    try:
        _get_token()
        return True, "✅ Connected to CJ Dropshipping"
    except CJError as e:
        return False, f"⚠️ {e}"
    except requests.RequestException as e:
        return False, f"⚠️ Could not reach CJ: {type(e).__name__}"


# ── Search ─────────────────────────────────────────────────────────────────

def _num(v):
    """CJ prices arrive as numbers, strings, or ranges like "0.57 -- 6.96"."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def product_url(pid, name=""):
    """CJ's own shape: /product/<slug>-p-<pid>.html."""
    slug = re.sub(r"[^a-z0-9.]+", "-", (name or "").lower()).strip("-.")
    return f"https://cjdropshipping.com/product/{slug + '-' if slug else ''}p-{pid}.html"


def normalize(p):
    pid = str(p.get("pid") or p.get("id") or "")
    name = p.get("productNameEn") or p.get("nameEn") or ""
    return {
        "id": pid,
        "name": name,
        "sku": p.get("productSku") or p.get("sku") or "",
        "image": p.get("productImage") or p.get("bigImage") or "",
        "price": p.get("sellPrice"),
        "price_min": _num(p.get("sellPrice")),
        "listed": _int(p.get("listedNum")),
        "created_ms": _int(p.get("createTime") or p.get("createAt")),
        "category": p.get("categoryName") or p.get("threeCategoryName") or "",
        "has_video": str(p.get("isVideo")) == "1",
        "free_shipping": bool(p.get("isFreeShipping")),
        "ships_from": p.get("shippingCountryCodes") or [],
        "url": product_url(pid, name),
    }


def _words(keyword):
    return [w for w in re.findall(r"[a-z0-9]+", (keyword or "").lower()) if w]


def matches(name, words):
    """Every keyword word must START a word in the name.

    A prefix, so "gun" finds "guns"; anchored, so "led" does not find "controlled".
    """
    low = (name or "").lower()
    return all(re.search(r"(?<![a-z0-9])" + re.escape(w), low) for w in words)


def search(keyword="", sort=DEFAULT_SORT, price_min=None, price_max=None,
           listed_min=None, listed_max=None, days=None, country="",
           free_shipping=False, video_only=False, pages=1):
    spec = SORTS.get(sort) or SORTS[DEFAULT_SORT]
    params = {"pageSize": PAGE_SIZE, "orderBy": spec["orderBy"], "sort": spec["sort"]}
    words = _words(keyword)
    if keyword.strip():
        params["productNameEn"] = keyword.strip()
    if listed_min not in (None, ""):
        params["minListedNum"] = _int(listed_min)
    if listed_max not in (None, ""):
        params["maxListedNum"] = _int(listed_max)
    if days:
        fmt = "%Y-%m-%d %H:%M:%S"
        params["createTimeFrom"] = time.strftime(fmt, time.gmtime(time.time() - int(days) * 86400))
        params["createTimeTo"] = time.strftime(fmt, time.gmtime())
    if country:
        params["countryCode"] = country
    if free_shipping:
        params["isFreeShipping"] = 1

    lo = _num(price_min) if price_min not in (None, "") else None
    hi = _num(price_max) if price_max not in (None, "") else None
    pages = max(1, min(MAX_PAGES, int(pages or 1)))

    found, seen, scanned, page = [], set(), 0, 0
    dropped = {"keyword": 0, "price": 0, "video": 0}
    for page in range(1, pages + 1):
        data = _get("/product/list", {**params, "pageNum": page}) or {}
        batch = data.get("list") or []
        scanned += len(batch)
        for raw in batch:
            p = normalize(raw)
            if not p["id"] or p["id"] in seen:
                continue
            seen.add(p["id"])
            if words and not matches(p["name"], words):
                dropped["keyword"] += 1
                continue
            if (lo is not None or hi is not None) and p["price_min"] is not None:
                if (lo is not None and p["price_min"] < lo) or (hi is not None and p["price_min"] > hi):
                    dropped["price"] += 1
                    continue
            if video_only and not p["has_video"]:
                dropped["video"] += 1
                continue
            found.append(p)
        if len(batch) < PAGE_SIZE:
            break

    return {"products": found, "scanned": scanned, "pages_scanned": page,
            "cj_total": _int(data.get("total")), "dropped": dropped,
            "more": scanned >= page * PAGE_SIZE}


def product_detail(pid):
    data = _get("/product/query", {"pid": pid, "features": "enable_video"}) or {}
    images = data.get("productImageSet") or []
    if isinstance(images, str):
        try:
            images = json.loads(images)
        except ValueError:
            images = [u for u in images.split(",") if u.startswith("http")]
    variants = [{"name": v.get("variantNameEn") or v.get("variantName") or v.get("variantKey") or "",
                 "sku": v.get("variantSku"), "price": v.get("variantSellPrice"),
                 "image": v.get("variantImage")}
                for v in (data.get("variants") or [])]
    name = data.get("productNameEn") or ""
    return {
        "id": pid,
        "name": name,
        "images": images[:12],
        "description": data.get("description") or "",
        "weight": data.get("productWeight"),
        "category": data.get("categoryName") or "",
        "price": data.get("sellPrice"),
        "listed": _int(data.get("listedNum")),
        "created": (data.get("createrTime") or "")[:10],
        "variants": variants[:40],
        "videos": data.get("productVideo") or [],
        "url": product_url(pid, name),
    }

import contextvars
import json
import os
import threading
import time

import requests

API_VERSION = "2024-04"

# ── Stores ─────────────────────────────────────────────────────────────────
# Main store = SHOPIFY_STORE / SHOPIFY_TOKEN (env). Extra stores come from env
# SHOPIFY_STORE_2 / SHOPIFY_TOKEN_2 / SHOPIFY_NAME_2 ... or are added in the UI
# (saved to stores.json on the persistent volume). A store can also use
# client_id + client_secret instead of a token (client-credentials grant,
# token refreshed automatically).
# Which store a request talks to is picked per browser (X-Store header) and
# kept in a context variable, so two devices on two stores do not mix.

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))
_STORES_FILE = os.path.join(_DATA_DIR, "stores.json")
_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ec4899", "#a855f7", "#14b8a6"]

_current = contextvars.ContextVar("shopify_store", default=None)
_lock = threading.Lock()
_cc_tokens = {}   # domain -> (token, expires_at)


def _norm_domain(d):
    d = (d or "").strip().lower()
    for p in ("https://", "http://"):
        if d.startswith(p):
            d = d[len(p):]
    d = d.split("/")[0]
    if d and "." not in d:
        d += ".myshopify.com"
    return d


def _env_stores():
    out = []
    main = _norm_domain(os.getenv("SHOPIFY_STORE", "hzw4be-qf.myshopify.com"))
    out.append({"domain": main, "token": os.getenv("SHOPIFY_TOKEN", ""),
                "name": os.getenv("SHOPIFY_NAME", "Bambyna"), "source": "env"})
    for i in range(2, 10):
        d = _norm_domain(os.getenv(f"SHOPIFY_STORE_{i}", ""))
        if not d:
            continue
        out.append({"domain": d, "token": os.getenv(f"SHOPIFY_TOKEN_{i}", ""),
                    "client_id": os.getenv(f"SHOPIFY_CLIENT_ID_{i}", ""),
                    "client_secret": os.getenv(f"SHOPIFY_CLIENT_SECRET_{i}", ""),
                    "name": os.getenv(f"SHOPIFY_NAME_{i}", d.split(".")[0]), "source": "env"})
    return out


def _file_stores():
    try:
        with open(_STORES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def stores():
    """All configured stores (env first, then ones added in the UI)."""
    seen, out = set(), []
    for s in _env_stores() + _file_stores():
        if s["domain"] in seen:
            continue
        seen.add(s["domain"])
        s = dict(s)
        s["color"] = _COLORS[len(out) % len(_COLORS)]
        out.append(s)
    return out


def public_stores():
    return [{"domain": s["domain"], "name": s.get("name") or s["domain"],
             "color": s["color"], "removable": s.get("source") != "env"} for s in stores()]


def default_domain():
    return stores()[0]["domain"]


def _store(domain=None):
    domain = _norm_domain(domain or _current.get() or "") or default_domain()
    for s in stores():
        if s["domain"] == domain:
            return s
    raise ValueError(f"Unknown store: {domain}")


def use_store(domain):
    """Set the store for the current request / thread context. Unknown -> main store."""
    d = _norm_domain(domain or "")
    if not d or not any(s["domain"] == d for s in stores()):
        d = default_domain()
    _current.set(d)
    return d


def current_store():
    s = _store()
    return {"domain": s["domain"], "name": s.get("name") or s["domain"], "color": s["color"]}


def _token(s):
    if s.get("token"):
        return s["token"]
    if s.get("client_id") and s.get("client_secret"):
        cached = _cc_tokens.get(s["domain"])
        if cached and cached[1] > time.time() + 300:
            return cached[0]
        r = requests.post(f"https://{s['domain']}/admin/oauth/access_token",
                          data={"grant_type": "client_credentials",
                                "client_id": s["client_id"],
                                "client_secret": s["client_secret"]}, timeout=30)
        r.raise_for_status()
        j = r.json()
        _cc_tokens[s["domain"]] = (j["access_token"], time.time() + int(j.get("expires_in", 86399)))
        return j["access_token"]
    return ""


def _base(domain=None):
    return f"https://{_store(domain)['domain']}/admin/api/{API_VERSION}"


def _headers(domain=None):
    return {"X-Shopify-Access-Token": _token(_store(domain)), "Content-Type": "application/json"}


def __getattr__(name):
    # Old code reads sc.SHOP / sc.HEADERS / sc.BASE / sc.TOKEN - resolve them for the current store.
    if name == "SHOP":
        return _store()["domain"]
    if name == "HEADERS":
        return _headers()
    if name == "BASE":
        return _base()
    if name == "TOKEN":
        return _token(_store())
    raise AttributeError(name)


def add_store(domain, name="", token="", client_id="", client_secret=""):
    """Check the credentials against /shop.json, then save the store. Returns its public info."""
    d = _norm_domain(domain)
    if not d.endswith(".myshopify.com"):
        raise ValueError("Use the store's .myshopify.com address (Settings -> Domains).")
    s = {"domain": d, "token": (token or "").strip(), "client_id": (client_id or "").strip(),
         "client_secret": (client_secret or "").strip()}
    if not s["token"] and not (s["client_id"] and s["client_secret"]):
        raise ValueError("Add an Admin API access token (shpat_...) or a Client ID + Client secret.")
    _cc_tokens.pop(d, None)
    try:
        tok = _token(s)
    except requests.HTTPError as e:
        raise ValueError(f"Could not get a token with that Client ID / secret: {e.response.text[:200]}")
    r = requests.get(f"https://{d}/admin/api/{API_VERSION}/shop.json",
                     headers={"X-Shopify-Access-Token": tok}, timeout=30)
    if r.status_code != 200:
        raise ValueError(f"Shopify said {r.status_code}: {r.text[:200]}")
    shop = r.json()["shop"]
    s["name"] = (name or "").strip() or shop.get("name") or d
    s["source"] = "file"
    with _lock:
        items = [x for x in _file_stores() if x["domain"] != d]
        items.append(s)
        with open(_STORES_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
    return next(p for p in public_stores() if p["domain"] == d)


def remove_store(domain):
    d = _norm_domain(domain)
    with _lock:
        items = [x for x in _file_stores() if x["domain"] != d]
        with open(_STORES_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
    _cc_tokens.pop(d, None)


# ── HTTP ───────────────────────────────────────────────────────────────────

def _get(path, params=None):
    r = requests.get(f"{_base()}{path}", headers=_headers(), params=params)
    r.raise_for_status()
    return r.json()


def _post(path, data):
    r = requests.post(f"{_base()}{path}", headers=_headers(), json=data)
    r.raise_for_status()
    return r.json()


def _put(path, data):
    r = requests.put(f"{_base()}{path}", headers=_headers(), json=data)
    r.raise_for_status()
    return r.json()


def _delete(path):
    r = requests.delete(f"{_base()}{path}", headers=_headers())
    r.raise_for_status()
    return {"success": True}


def get_shop():
    return _get("/shop.json")["shop"]


def list_products(limit=50, title=None):
    params = {"limit": limit}
    if title:
        params["title"] = title
    return _get("/products.json", params)["products"]


def get_product(product_id):
    return _get(f"/products/{product_id}.json")["product"]


def create_product(title, body_html="", vendor="", product_type="", tags="",
                   price="0.00", compare_at_price=None, sku="", quantity=0,
                   images=None, variants=None):
    variant = {"price": price, "sku": sku, "inventory_quantity": quantity}
    if compare_at_price:
        variant["compare_at_price"] = compare_at_price
    product = {
        "title": title,
        "body_html": body_html,
        "vendor": vendor,
        "product_type": product_type,
        "tags": tags,
        "variants": variants or [variant],
    }
    if images:
        product["images"] = [{"src": img} if isinstance(img, str) else img for img in images]
    return _post("/products.json", {"product": product})["product"]


def update_product(product_id, **kwargs):
    return _put(f"/products/{product_id}.json", {"product": kwargs})["product"]


def delete_product(product_id):
    return _delete(f"/products/{product_id}.json")


def list_collections():
    customs = _get("/custom_collections.json")["custom_collections"]
    smarts = _get("/smart_collections.json")["smart_collections"]
    return customs + smarts


def create_collection(title, body_html=""):
    return _post("/custom_collections.json",
                 {"custom_collection": {"title": title, "body_html": body_html}})["custom_collection"]


def list_orders(limit=50, status="any"):
    return _get("/orders.json", {"limit": limit, "status": status})["orders"]


def get_order(order_id):
    return _get(f"/orders/{order_id}.json")["order"]


def list_pages(limit=50):
    return _get("/pages.json", {"limit": limit})["pages"]


def create_page(title, body_html=""):
    return _post("/pages.json", {"page": {"title": title, "body_html": body_html}})["page"]


def get_inventory_levels(location_id=None):
    params = {}
    if location_id:
        params["location_ids"] = location_id
    return _get("/inventory_levels.json", params).get("inventory_levels", [])


def list_locations():
    return _get("/locations.json")["locations"]


# ── Theme ──────────────────────────────────────────────────────────────────

def get_active_theme():
    themes = _get("/themes.json")["themes"]
    for t in themes:
        if t.get("role") == "main":
            return t
    return themes[0] if themes else None


def list_theme_files(theme_id):
    return _get(f"/themes/{theme_id}/assets.json")["assets"]


def get_theme_file(theme_id, key):
    data = _get(f"/themes/{theme_id}/assets.json", {"asset[key]": key, "theme_id": theme_id})
    return data["asset"]


def update_theme_file(theme_id, key, value):
    return _put(f"/themes/{theme_id}/assets.json",
                {"asset": {"key": key, "value": value}})["asset"]


def delete_theme_file(theme_id, key):
    r = requests.delete(f"{_base()}/themes/{theme_id}/assets.json",
                        headers=_headers(), params={"asset[key]": key})
    r.raise_for_status()
    return {"success": True}

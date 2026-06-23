import requests
import os

SHOP = os.getenv("SHOPIFY_STORE", "hzw4be-qf.myshopify.com")
TOKEN = os.getenv("SHOPIFY_TOKEN", "")
BASE = f"https://{SHOP}/admin/api/2024-04"
HEADERS = {"X-Shopify-Access-Token": TOKEN, "Content-Type": "application/json"}


def _get(path, params=None):
    r = requests.get(f"{BASE}{path}", headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


def _post(path, data):
    r = requests.post(f"{BASE}{path}", headers=HEADERS, json=data)
    r.raise_for_status()
    return r.json()


def _put(path, data):
    r = requests.put(f"{BASE}{path}", headers=HEADERS, json=data)
    r.raise_for_status()
    return r.json()


def _delete(path):
    r = requests.delete(f"{BASE}{path}", headers=HEADERS)
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

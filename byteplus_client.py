"""BytePlus ModelArk client — the ARK counterpart to `fal_client`.

Same job as fal_client, different service, and the differences are not
cosmetic:

  * There is no file storage. fal hands you an upload URL and you pass a link;
    ARK takes the image inline as a `data:` URI. That removes a network round
    trip and a failure mode, at the cost of a 64 MB ceiling on the whole
    request body.
  * The queue is one endpoint, not three. You POST a task, you GET the task,
    and the same object carries the status, the video URL, the settings the
    model actually used, and the token count you are billed on.
  * `usage.completion_tokens` on a finished task is the authoritative charge.
    Everything this app quotes beforehand is an estimate off the published
    formula; this number is what BytePlus actually bills, so it is read back
    and reported rather than assumed.

Region: the key is issued per region and the host has to match it. This is the
ap-southeast-1 host, which is the one BytePlus documents for international
accounts.
"""

import os
import time
import base64

import requests

BASE = "https://ark.ap-southeast.bytepluses.com/api/v3"
TASKS = BASE + "/contents/generations/tasks"


class ArkError(Exception):
    pass


def _key():
    key = os.getenv("ARK_API_KEY", "").strip()
    if not key:
        raise ArkError("ARK_API_KEY is not set. Add it to .env locally and to "
                       "Railway Variables.")
    return key


def _headers():
    return {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}


def _fail(what, r):
    """One error shape, with the body — ARK explains itself in the body.

    A rejected task comes back as {"error": {"code": ..., "message": ...}}, and
    the message names the parameter that was wrong. Truncating it to the status
    code would throw away the only part worth reading.
    """
    detail = r.text[:500]
    try:
        err = (r.json() or {}).get("error") or {}
        if err.get("message"):
            detail = "%s (%s)" % (err["message"], err.get("code") or r.status_code)
    except Exception:
        pass
    raise ArkError(f"BytePlus {what} failed ({r.status_code}): {detail}")


# ── Queue plumbing ─────────────────────────────────────────────────────────

def submit(payload):
    """Create a video generation task. Returns the task id."""
    r = requests.post(TASKS, headers=_headers(), json=payload, timeout=120)
    if r.status_code >= 400:
        _fail("task creation", r)
    task_id = (r.json() or {}).get("id")
    if not task_id:
        raise ArkError(f"BytePlus accepted the task but returned no id: {r.text[:300]}")
    return task_id


def get_task(task_id):
    r = requests.get(f"{TASKS}/{task_id}",
                     headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    if r.status_code >= 400:
        _fail("task lookup", r)
    return r.json()


def run(payload, on_status=None, timeout=1800, poll_every=5):
    """Submit and block until the task finishes. Returns the finished task.

    Seedance validates twice: once when the task is created, and again once a
    worker picks it up. The second one is why a task can sit in `queued` and
    then fail with a parameter error — see the task-type constraints in
    `dreamina_studio`. Both arrive here as an ArkError with the model's own
    wording, which is more useful than anything this layer could invent.
    """
    task_id = submit(payload)
    if on_status:
        on_status(f"queued ({task_id[-8:]})")

    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        time.sleep(poll_every)
        task = get_task(task_id)
        state = task.get("status")
        if state != last:
            last = state
            if on_status:
                on_status(state or "unknown")
        if state == "succeeded":
            return task
        if state in ("failed", "cancelled", "expired"):
            err = task.get("error") or {}
            raise ArkError("BytePlus task %s: %s" % (
                state, err.get("message") or str(task)[:300]))
    raise ArkError(f"BytePlus task timed out after {timeout}s (id {task_id})")


def list_tasks(page_size=10):
    r = requests.get(TASKS, params={"page_size": page_size},
                     headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    if r.status_code >= 400:
        _fail("task list", r)
    return r.json()


# ── File input ─────────────────────────────────────────────────────────────

def to_data_uri(data, content_type="image/jpeg"):
    """ARK's only inline input format for an image.

    Format matters: the subtype must be lowercase, because ARK parses it out of
    the URI rather than sniffing the bytes.
    """
    return f"data:{content_type.lower()};base64,{base64.b64encode(data).decode()}"


def download(url, timeout=600):
    """Fetch a finished video.

    The URL is signed and lives 24 hours, with a cap of 100 downloads, so the
    file is pulled to disk once, right after generation, and served locally
    from there.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


# ── Account ────────────────────────────────────────────────────────────────

def check_account():
    """Returns (ok, human-readable message).

    Listing tasks is the free way to prove the key works. There is no balance
    endpoint on ARK the way there is on fal, so this can confirm the key and
    the region but not that the account has money in it — an empty account
    fails at task creation instead, with a message that says so.
    """
    try:
        data = list_tasks(page_size=1)
    except ArkError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Could not reach BytePlus: {e}"

    total = data.get("total")
    if isinstance(total, int) and total:
        return True, f"BytePlus ModelArk connected — {total} task(s) in the last 7 days"
    return True, "BytePlus ModelArk connected"

from __future__ import annotations
import hashlib
import json
import os
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ---------------- Config ----------------
PROVIDER = os.getenv("AERO_SEARCH_PROVIDER", "bing").lower()   # "bing" | "serpapi" | "none"
BING_KEY = os.getenv("AERO_BING_KEY", "")
SERP_KEY = os.getenv("SERPAPI_KEY", "")
MAX_RESULTS = int(os.getenv("AERO_WEB_MAX_RESULTS", "12"))
MKT = os.getenv("AERO_WEB_MKT", "en-US")

# Prefer authoritative domains; set empty to disable filtering.
DEFAULT_ALLOW = [
    "faa.gov", "nasa.gov", "esa.int", "jaxa.jp", "dlr.de", "spaceforce.mil",
    "boeing.com", "spacex.com", "rocketlabusa.com"
]
ALLOWLIST = [d.strip().lower() for d in os.getenv(
    "AERO_WEB_DOMAINS_ALLOW", ",".join(DEFAULT_ALLOW)
).split(",") if d.strip()]

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache", "web"))
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_S = int(os.getenv("AERO_WEB_CACHE_TTL_S", "86400"))  # 1 day

UA = {"User-Agent": "AeroAI/1.0 (+https://example.com)"}

# ---------------- Cache helpers ----------------
def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def _cache_get(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - data.get("_ts", 0) > CACHE_TTL_S:
            return None
        return data.get("payload")
    except Exception:
        return None

def _cache_set(key: str, payload: dict) -> None:
    p = _cache_path(key)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"_ts": time.time(), "payload": payload}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------- Fetch + extract ----------------
def _fetch(url: str, timeout: int = 10) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return None
    return None

def _extract_text(html: str, max_chars: int = 1000) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception:
        return html[:max_chars]

def _domain_ok(url: str) -> bool:
    if not ALLOWLIST:
        return True
    try:
        host = (urlparse(url).netloc or "").lower()
        return any(host.endswith(d) for d in ALLOWLIST)
    except Exception:
        return True

def _dedupe(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for it in items:
        u = it.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out

# ---------------- Search providers ----------------
def _search_bing(q: str, count: int) -> List[dict]:
    if not BING_KEY:
        return []
    key = f"bing::{q}::{count}::{MKT}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {"q": q, "count": count, "mkt": MKT, "responseFilter": "Webpages"}
    try:
        r = requests.get(url, params=params, headers={"Ocp-Apim-Subscription-Key": BING_KEY, **UA}, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = []
        for v in (data.get("webPages") or {}).get("value", []):
            items.append({"title": v.get("name"), "url": v.get("url"), "snippet": v.get("snippet")})
        _cache_set(key, items)
        return items
    except Exception:
        return []

def _search_serpapi(q: str, count: int) -> List[dict]:
    if not SERP_KEY:
        return []
    key = f"serpapi::{q}::{count}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": q, "num": min(count, 10), "api_key": SERP_KEY}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = []
        for v in data.get("organic_results", []):
            items.append({"title": v.get("title"), "url": v.get("link"), "snippet": v.get("snippet")})
        _cache_set(key, items)
        return items
    except Exception:
        return []

def search_web(q: str, count: int = 6) -> List[dict]:
    if PROVIDER == "none":
        return []
    if PROVIDER == "serpapi":
        items = _search_serpapi(q, count)
    else:
        items = _search_bing(q, count)
    allow = [x for x in items if x.get("url") and _domain_ok(x["url"])]
    other = [x for x in items if x not in allow]
    return _dedupe(allow + other)

# ---------------- Query planning + collection ----------------
def _queries(origin_hint: Optional[str], target: Optional[str]) -> List[str]:
    q = [
        "site:faa.gov spaceports",
        "site:faa.gov licensed commercial spaceports list",
        "site:nasa.gov Artemis landing regions",
        "site:nasa.gov lunar south pole landing sites overview"
    ]
    if origin_hint:
        q.append(f"nearest orbital launch site to {origin_hint}")
        q.append(f"spaceport near {origin_hint}")
    if target:
        q.append(f"best US launch sites for {target} mission")
    return q

def collect_web_snippets(origin_hint: Optional[str], target: Optional[str], max_total: int = MAX_RESULTS) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for q in _queries(origin_hint, target):
        hits = search_web(q, count=max(4, min(8, max_total)))
        for h in hits:
            url = h.get("url")
            if not url:
                continue
            html = _fetch(url)
            if not html:
                continue
            excerpt = _extract_text(html, max_chars=800)
            results.append({
                "title": h.get("title") or "Untitled",
                "url": url,
                "query": q,
                "excerpt": excerpt
            })
            if len(results) >= max_total:
                break
        if len(results) >= max_total:
            break
    seen = set()
    out: List[Dict[str, str]] = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        out.append(r)
    return out

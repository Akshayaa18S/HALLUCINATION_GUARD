"""
Run this directly to see EXACTLY what our WikipediaRetriever does and
why it's coming back empty - prints the raw JSON at each step instead
of swallowing it.

Usage (from the backend/ directory, with your venv activated):
    python diagnose_wikipedia.py "Primates"
    python diagnose_wikipedia.py "Monkeys are primates"
"""

import asyncio
import sys

import httpx

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "HallucinationDetectorBackend/1.0 "
        "(https://github.com/example/hallucination-detector; contact@example.com) "
        "python-httpx"
    )
}


async def fetch_extract_by_title(title: str):
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
        resp = await client.get(WIKI_API, params=params)
        print(f"  [title lookup] GET {resp.request.url}")
        print(f"  [title lookup] status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        print(f"  [title lookup] raw response: {data}")
        return data


async def search_best_title(query: str):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
        resp = await client.get(WIKI_API, params=params)
        print(f"  [search] GET {resp.request.url}")
        print(f"  [search] status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        print(f"  [search] raw response: {data}")
        return data


async def main(query_text: str):
    print(f"=== Step 1: exact-title lookup for {query_text!r} ===")
    data = await fetch_extract_by_title(query_text)
    pages = data.get("query", {}).get("pages", {})
    found_extract = False
    for page_id, page in pages.items():
        if page_id == "-1" or "extract" not in page:
            print(f"  -> page_id={page_id}: NO 'extract' key present. Page dict: {page}")
            continue
        extract = page["extract"].strip()
        if not extract:
            print(f"  -> page_id={page_id}: 'extract' key present but EMPTY string")
            continue
        print(f"  -> SUCCESS: title={page.get('title')!r}, extract (first 200 chars): {extract[:200]!r}")
        found_extract = True

    if found_extract:
        print("\nStep 1 succeeded - search fallback not needed.")
        return

    print(f"\n=== Step 2: search fallback for {query_text!r} ===")
    search_data = await search_best_title(query_text)
    results = search_data.get("query", {}).get("search", [])
    if not results:
        print("  -> NO search results at all.")
        return
    best_title = results[0]["title"]
    print(f"  -> best matching title: {best_title!r}")

    print(f"\n=== Step 3: exact-title lookup for search result {best_title!r} ===")
    data2 = await fetch_extract_by_title(best_title)
    pages2 = data2.get("query", {}).get("pages", {})
    for page_id, page in pages2.items():
        if page_id == "-1" or "extract" not in page:
            print(f"  -> page_id={page_id}: NO 'extract' key present. Page dict: {page}")
            continue
        extract = page["extract"].strip()
        if not extract:
            print(f"  -> page_id={page_id}: 'extract' key present but EMPTY string")
            continue
        print(f"  -> SUCCESS: title={page.get('title')!r}, extract (first 200 chars): {extract[:200]!r}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Primates"
    asyncio.run(main(query))

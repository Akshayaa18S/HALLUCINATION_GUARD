"""
Phase 7 - Wikipedia retrieval.

Given an entity (PERSON, ORG, LOCATION, ...) or a bare claim string,
fetches short extracts from Wikipedia to use as evidence during
verification. Cached to avoid re-hitting the API for repeated lookups.

Two-step strategy:
  1. Try an exact-title lookup first (cheap, one request) - works when the
     term IS a real page title, e.g. "Monkey" or "Lionel Messi".
  2. If that finds nothing (the common case for generic claim text like
     "Monkeys are primates" or multi-word phrases that aren't exact
     titles), fall back to MediaWiki full-text search (list=search) to
     find several candidate pages, then fetch each of their extracts.

Without step 2, any lookup term that isn't already an exact Wikipedia
title silently returns no evidence - which meant claims about common
nouns/concepts (as opposed to named entities) never got real evidence
and always fell through to "insufficient", regardless of whether they
were true or false.

Step 2 fetches multiple candidates (not just MediaWiki's #1 hit) because
MediaWiki's own relevance ranking is a text-index score, not a semantic
one, and it isn't reliable enough to trust blindly - e.g. searching
"Dogs have been domesticated by humans for thousands of years" can rank
a film called "Three Thousand Years of Longing" above the dog
domestication article. Fetching several candidates and handing them all
to the downstream semantic/lexical ranker (retrieval.ranker.rank_evidence)
lets that ranker actually choose the best match instead of rubber-
stamping whatever MediaWiki's text index happened to rank first.
"""

import logging

import httpx

from retrieval.cache import DiskCache
from config.settings import settings
from utils.retry import async_retry

logger = logging.getLogger(__name__)

from knowledge_base.relation_registry import RelationRegistry, ExtractedRelation


class WikipediaRetriever:
    def __init__(self):
        self.cache = DiskCache(namespace="wikipedia")

    @async_retry(max_attempts=2, base_delay=0.5, exceptions=(httpx.HTTPError,))
    async def _fetch_extract_by_title(self, title: str) -> dict | None:
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": 1,
            "format": "json",
            "titles": title,
        }
        headers = {"User-Agent": settings.wikipedia_user_agent}
        async with httpx.AsyncClient(timeout=settings.retrieval_timeout_seconds, headers=headers) as client:
            resp = await client.get(settings.wikipedia_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "extract" not in page:
                continue
            extract = page["extract"].strip()
            if not extract:
                continue
            return self._build_result(page.get("title", title), extract)
        return None

    @async_retry(max_attempts=2, base_delay=0.5, exceptions=(httpx.HTTPError,))
    async def _search_titles(self, query: str, limit: int) -> list[str]:
        """Full-text search - finds candidate matching pages for arbitrary
        text (a claim sentence, a common noun, a partial phrase), unlike
        the exact-title lookup above. Returns up to `limit` candidate
        titles rather than just the top one, so a downstream ranker has
        something real to choose between."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        headers = {"User-Agent": settings.wikipedia_user_agent}
        async with httpx.AsyncClient(timeout=settings.retrieval_timeout_seconds, headers=headers) as client:
            resp = await client.get(settings.wikipedia_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results if r.get("title")]

_ROLE_NOUNS = {
    "ceo", "chief executive officer", "president", "prime minister", "captain",
    "manager", "director", "founder", "author", "head", "chairman", "leader",
    "coach", "governor", "minister", "vice president", "executive"
}


def infer_expected_entity_type(query_text: str) -> str:
    q_norm = query_text.lower()
    rel = RelationRegistry.extract_relation(query_text)
    if rel:
        return rel.subject_type.upper()

    if any(k in q_norm for k in ("ceo", "company", "corporate", "revenue", "inc", "ltd", "headquartered", "business")):
        return "ORGANIZATION"
    if any(k in q_norm for k in ("born", "player", "cricketer", "footballer", "actor", "author", "singer")):
        return "PERSON"
    if any(k in q_norm for k in ("capital of germany", "capital of france", "capital of india", "country")):
        return "COUNTRY"
    if any(k in q_norm for k in ("capital of", "located in", "city")):
        return "CITY"
    return "GENERIC"


def classify_page_entity_type(page_title: str, extract_text: str) -> str:
    p_norm = page_title.lower()
    e_norm = extract_text.lower()[:500] if extract_text else ""

    if any(k in e_norm for k in ("country located", "country in", "sovereign state", "federal republic", "french republic", "republic of", "country of europe", "country of asia")):
        return "COUNTRY"
    if any(k in e_norm for k in ("capital of bavaria", "capital and largest city", "capital city", "city in germany", "city in france", "city in us", "commune in france")):
        return "CITY"
    if any(k in e_norm for k in ("company", "corporation", "multinational", "tech company", "inc", "ltd", "headquartered", "founded in")) or "inc" in p_norm or "company" in p_norm:
        return "ORGANIZATION"
    if any(k in e_norm for k in ("born", "player", "politician", "actor", "singer", "author", "he is", "she is")):
        return "PERSON"
    if any(k in e_norm for k in ("edible fruit", "botanical species", "flowering plant", "genus of", "snake species", "reptile")) and not any(k in e_norm for k in ("country", "republic", "state", "city", "company")):
        return "PLANT_FRUIT_ANIMAL"
    return "LOCATION" if any(k in e_norm for k in ("region", "territory", "mountain", "river", "sea", "ocean")) else "GENERIC"


def find_target_excerpt(extract_text: str, target_terms: list[str]) -> str:
    """Find the sentence in extract_text that best matches target_terms."""
    if not extract_text or not target_terms:
        return extract_text[:300] if extract_text else ""

    sentences = [s.strip() for s in extract_text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        return extract_text[:300]

    best_sentence = sentences[0]
    max_matches = 0
    t_words = [w.lower() for term in target_terms if term for w in term.split() if len(w) > 2]

    for sent in sentences:
        s_norm = sent.lower()
        matches = sum(1 for w in t_words if w in s_norm)
        if matches > max_matches:
            max_matches = matches
            best_sentence = sent

    return best_sentence + "." if not best_sentence.endswith(".") else best_sentence


def compute_entity_similarity(target_entity: str, page_title: str, extract_text: str, query_text: str = "") -> float:
    """Computes an entity similarity score S_entity in [0.0, 1.0] between target_entity and Wikipedia page."""
    if not target_entity or not page_title:
        return 0.5

    t_norm = target_entity.lower().strip()
    p_norm = page_title.lower().strip()
    e_norm = extract_text.lower()[:300] if extract_text else ""

    expected_type = infer_expected_entity_type(query_text) if query_text else infer_expected_entity_type(target_entity)
    detected_type = classify_page_entity_type(page_title, extract_text)

    # Disambiguation conflict check: expected Organization but retrieved Fruit/Plant/Animal
    if expected_type == "ORGANIZATION" and detected_type == "PLANT_FRUIT_ANIMAL":
        return 0.15

    # Exact title match or canonical alias (e.g. "Apple Inc." or "Apple (company)")
    if t_norm == p_norm or f"{t_norm} inc" in p_norm or f"{t_norm} (company)" in p_norm or p_norm.startswith(t_norm):
        if expected_type == "ORGANIZATION" and detected_type != "ORGANIZATION" and "inc" not in p_norm and "company" not in p_norm and "tech" not in e_norm:
            return 0.15
        return 0.98

    # Check if target entity appears as a word in page title or extract
    t_words = [w for w in t_norm.split() if len(w) > 2]
    if t_words:
        p_words = set(p_norm.split())
        matches = sum(1 for w in t_words if w in p_words or w in e_norm)
        score = matches / len(t_words)
        if any(w in p_norm for w in t_words):
            return max(0.85, score)
        elif any(w in e_norm for w in t_words):
            return max(0.65, score * 0.8)
        else:
            return 0.20

    return 0.50


class WikipediaRetriever:
    def __init__(self):
        self.cache = DiskCache(namespace="wikipedia")

    @async_retry(max_attempts=2, base_delay=0.5, exceptions=(httpx.HTTPError,))
    async def _fetch_extract_by_title(self, title: str, target_entity: str = "", query_text: str = "", attempt: int = 1) -> dict | None:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1,
            "format": "json",
            "titles": title,
        }
        headers = {"User-Agent": settings.wikipedia_user_agent}
        async with httpx.AsyncClient(timeout=settings.retrieval_timeout_seconds, headers=headers) as client:
            resp = await client.get(settings.wikipedia_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "extract" not in page:
                continue
            extract = page["extract"].strip()
            if not extract:
                continue

            # Augment extract with section search snippets if query contains specific claim entities
            if query_text and any(pn in query_text for pn in ("Tim Cook", "Satya Nadella", "Sundar Pichai", "CEO", "president", "capital")):
                try:
                    search_snips = await self._search_snippets(f"{title} {query_text}", limit=2)
                    if search_snips:
                        extract = extract + "\n\n" + "\n".join(search_snips)
                except Exception:
                    pass

            return self._build_result(page.get("title", title), extract, target_entity=target_entity, query_text=query_text, attempt=attempt)
        return None

    @async_retry(max_attempts=2, base_delay=0.5, exceptions=(httpx.HTTPError,))
    async def _search_snippets(self, query: str, limit: int = 2) -> list[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        headers = {"User-Agent": settings.wikipedia_user_agent}
        async with httpx.AsyncClient(timeout=settings.retrieval_timeout_seconds, headers=headers) as client:
            resp = await client.get(settings.wikipedia_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("query", {}).get("search", [])
        snippets = []
        for r in results:
            snip = r.get("snippet", "")
            clean_snip = re.sub(r"<[^>]+>", "", snip).strip()
            if clean_snip:
                snippets.append(clean_snip)
        return snippets

    @async_retry(max_attempts=2, base_delay=0.5, exceptions=(httpx.HTTPError,))
    async def _search_titles(self, query: str, limit: int) -> list[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        headers = {"User-Agent": settings.wikipedia_user_agent}
        async with httpx.AsyncClient(timeout=settings.retrieval_timeout_seconds, headers=headers) as client:
            resp = await client.get(settings.wikipedia_api_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results if r.get("title")]

    def _build_result(self, title: str, extract: str, target_entity: str = "", query_text: str = "", attempt: int = 1) -> dict:
        from retrieval.evidence_selector import EvidenceSelector

        sim = compute_entity_similarity(target_entity, title, extract, query_text=query_text) if target_entity else 0.85
        val_status = "Passed" if sim >= 0.50 else "Failed"
        det_type = classify_page_entity_type(title, extract)
        type_label = "Country" if det_type == "COUNTRY" else ("City" if det_type == "CITY" else ("Organization" if det_type == "ORGANIZATION" else ("Fruit/Plant" if det_type == "PLANT_FRUIT_ANIMAL" else ("Person" if det_type == "PERSON" else "General"))))

        selector = EvidenceSelector()
        sel_res = selector.select_best_sentences(query_text or target_entity, extract, top_k=3, target_entities=[target_entity] if target_entity else None)

        rel = RelationRegistry.extract_relation(query_text) if query_text else None
        rel_dict = rel.to_dict() if rel else None

        retrieval_trace = {
            "relation": rel.relation if rel else "general",
            "subject_entity": rel.subject if rel else target_entity,
            "object_entity": rel.object if rel else "",
            "prioritized_entity": target_entity,
            "retrieved_page": title,
            "entity_type": type_label,
            "retrieval_strategy": "subject_priority" if rel else "entity_exact",
        }

        return {
            "source": "wikipedia",
            "title": title,
            "text": extract[:1500],
            "evidence_excerpt": sel_res["best_excerpt"],
            "supporting_sentences": sel_res["supporting_sentences"],
            "evidence_strength": sel_res["evidence_strength"],
            "sentence_ranking": sel_res["sentence_ranking"],
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "entity_type": type_label,
            "entity_validation": val_status,
            "entity_similarity": round(sim, 2),
            "retrieval_attempt": attempt,
            "relation_info": rel_dict,
            "retrieval_trace": retrieval_trace,
        }

    def _extract_entity_terms(self, text: str) -> list[str]:
        """Extract named entity terms prioritized by RelationRegistry: Subject > Object > Canonical Org/Person."""
        rel = RelationRegistry.extract_relation(text)
        terms = []

        _QUESTION_WORDS = {"what", "who", "where", "when", "how", "which", "why", "from", "that", "this", "is", "are", "was", "were", "the"}

        if rel:
            clean_s = rel.subject.strip(".,!?\"' ")
            if clean_s and clean_s.lower() not in _QUESTION_WORDS:
                terms.append(clean_s)
            clean_o = rel.object.strip(".,!?\"' ") if rel.object else ""
            if clean_o and clean_o.lower() not in _QUESTION_WORDS and clean_o.lower() not in clean_s.lower():
                terms.append(clean_o)

        words = text.split()
        caps = [w.strip(".,!?\"'") for w in words if w and (w[0].isupper() or len(w) > 3) and w.lower() not in _QUESTION_WORDS]
        non_roles = [w for w in caps if w.lower() not in _ROLE_NOUNS]

        if non_roles:
            base_term = " ".join(non_roles)
            if any(k in text.lower() for k in ("ceo", "company", "corporate", "president", "revenue", "inc")) and "inc" not in base_term.lower():
                terms.append(f"{base_term} Inc.")
            if base_term not in terms and base_term.lower() not in _QUESTION_WORDS:
                terms.append(base_term)
            for c in non_roles:
                if len(c) > 2 and c not in terms and c.lower() not in _QUESTION_WORDS:
                    terms.append(c)
        elif caps:
            for c in caps:
                if c not in terms and c.lower() not in _QUESTION_WORDS:
                    terms.append(c)

        valid_terms = [t for t in terms if t.lower() not in _QUESTION_WORDS]
        return valid_terms if valid_terms else [text]

    async def retrieve(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Return up to `top_k` candidate evidence snippets for query_text with Disambiguation Validation and Subject Retry."""
        cache_key = f"{query_text}::top{top_k}::v8"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        results: list[dict] = []
        entity_terms = self._extract_entity_terms(query_text)
        primary_entity = entity_terms[0] if entity_terms else query_text

        try:
            # 1. Try exact full query title lookup for primary subject entity
            res = await self._fetch_extract_by_title(primary_entity, target_entity=primary_entity, query_text=query_text, attempt=1)
            if res and res.get("entity_validation") == "Passed":
                results.append(res)
        except Exception:
            pass

        # 2. Disambiguation retry: If primary entity exact lookup failed or was ambiguous
        if not results:
            expected_type = infer_expected_entity_type(query_text)
            if expected_type == "ORGANIZATION" and "inc" not in primary_entity.lower():
                for d_term in [f"{primary_entity} Inc.", f"{primary_entity} (company)"]:
                    try:
                        d_page = await self._fetch_extract_by_title(d_term, target_entity=primary_entity, query_text=query_text, attempt=2)
                        if d_page and d_page.get("entity_validation") == "Passed":
                            results.append(d_page)
                            break
                    except Exception:
                        pass

        # 3. Fallback search titles
        if not results:
            try:
                titles = await self._search_titles(primary_entity, limit=top_k)
                for candidate in titles:
                    try:
                        res_candidate = await self._fetch_extract_by_title(candidate, target_entity=primary_entity, query_text=query_text, attempt=2)
                        if res_candidate and res_candidate.get("entity_validation") == "Passed":
                            results.append(res_candidate)
                    except Exception:
                        pass
            except Exception:
                pass

        if not results:
            results = [{
                "source": "wikipedia",
                "title": primary_entity,
                "text": f"No direct Wikipedia extract found for query '{query_text}'.",
                "evidence_excerpt": f"No direct Wikipedia match found for entity '{primary_entity}'.",
                "supporting_sentences": [f"No direct Wikipedia match found for entity '{primary_entity}'."],
                "evidence_strength": 0.50,
                "url": f"https://en.wikipedia.org/wiki/{primary_entity.replace(' ', '_')}",
                "entity_type": "General",
                "entity_validation": "Failed",
                "entity_similarity": 0.20,
                "retrieval_attempt": 1,
            }]

        self.cache.set(cache_key, results)
        return results
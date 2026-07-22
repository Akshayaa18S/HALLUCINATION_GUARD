"""Real evidence retrieval for Stage 6 (Fact Verification).

Replaces the old two-topic hardcoded document list with:

  1. A **static** corpus (FEVER + HaluEval seed facts, see
     ``data/knowledge/*.jsonl``) that is embedded and indexed with FAISS
     exactly once, then persisted to ``data/index/static.faiss`` and reused
     on every subsequent call/process start (``load()`` is idempotent).
  2. **Live Wikipedia retrieval** via the ``wikipedia`` package, so queries
     about topics that aren't in the static seed set still get real,
     on-topic evidence instead of silently falling back to unrelated facts.

``KnowledgeBase.retrieve(query, k)`` merges both sources, re-scores every
candidate against the query with the shared embedding model, deduplicates,
and returns the top ``k`` ``(RetrievalDocument, score)`` pairs — the same
contract ``Retriever.retrieve_top_k`` already exposes, so callers (e.g.
``verification_service``) don't need to know which source a document came
from.

Nothing here fabricates facts or sources: if neither the static corpus nor
Wikipedia can produce evidence for a query, ``retrieve`` returns an empty
list and callers must say so explicitly (see
``verification_service.VerificationService.verify``).
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlencode
from typing import List, Optional, Tuple

import numpy as np

from services.embedding_service import embedding_service
from services.entity_recognition import NamedEntity, entity_recognizer
from services.retriever import RetrievalDocument, Retriever

logger = logging.getLogger("hallucination_guard.knowledge_base")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_DIR / "data" / "knowledge"
INDEX_DIR = _BACKEND_DIR / "data" / "index"
STATIC_INDEX_PATH = INDEX_DIR / "static.faiss"

# Wikipedia title aliases belong to the live-source adapter, not the public API
# or the semantic retrieval layer. Values are canonical lookup titles.
_WIKIPEDIA_ALIASES = {
    "bts": "Bangtan Sonyeondan",
    "mbappe": "Kylian Mbapp\u00e9",
    "kylian mbappe": "Kylian Mbapp\u00e9",
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "kohli": "Virat Kohli",
}

_WIKIPEDIA_REST_BASE = "https://en.wikipedia.org/api/rest_v1"
_WIKIPEDIA_SEARCH_BASE = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_WIKIPEDIA_USER_AGENT = "HallucinationGuard/1.0 (Wikipedia evidence retrieval)"
_WIKIPEDIA_RETRIES = 3
FEVER_SIMILARITY_THRESHOLD = 0.55
FEVER_SIMILARITY_THRESHOLD = 0.55


class _WikipediaRestClient:
    """Defensive MediaWiki REST client used when ``wikipedia`` cannot parse a response."""

    def get_json(self, url: str) -> Optional[dict]:
        headers = {"Accept": "application/json", "User-Agent": _WIKIPEDIA_USER_AGENT}
        for attempt in range(1, _WIKIPEDIA_RETRIES + 1):
            try:
                request = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(request, timeout=10) as response:
                    status = getattr(response, "status", response.getcode())
                    response_headers = dict(response.headers.items())
                    body = response.read().decode("utf-8", errors="replace")
                    self._log_response(url, response.geturl(), status, response_headers, body)
                    return self._parse_json(url, status, response_headers, body)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                response_headers = dict(exc.headers.items()) if exc.headers else {}
                self._log_response(url, exc.geturl(), exc.code, response_headers, body)
                if not self._is_transient_status(exc.code) or attempt == _WIKIPEDIA_RETRIES:
                    logger.warning("Wikipedia REST request failed permanently: url=%s status=%s", url, exc.code)
                    return None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                logger.warning("Wikipedia REST transport error: url=%s attempt=%s/%s error=%r", url, attempt, _WIKIPEDIA_RETRIES, exc)
                if attempt == _WIKIPEDIA_RETRIES:
                    return None
            if attempt < _WIKIPEDIA_RETRIES:
                delay = 0.5 * (2 ** (attempt - 1))
                logger.info("Retrying Wikipedia REST request: url=%s delay_seconds=%.1f", url, delay)
                time.sleep(delay)
        return None

    @staticmethod
    def _is_transient_status(status: int) -> bool:
        return status in {408, 425, 429} or status >= 500

    @staticmethod
    def _log_response(request_url: str, response_url: str, status: int, headers: dict, body: str) -> None:
        logger.info(
            "Wikipedia REST response: request_url=%s response_url=%s status=%s headers=%s body_first_200=%r",
            request_url, response_url, status, headers, body[:200],
        )
        if response_url != request_url:
            logger.warning("Wikipedia REST redirect detected: request_url=%s response_url=%s", request_url, response_url)

    @staticmethod
    def _parse_json(url: str, status: int, headers: dict, body: str) -> Optional[dict]:
        content_type = headers.get("Content-Type", headers.get("content-type", "")).lower()
        if not body.strip():
            logger.warning("Wikipedia REST returned an empty body: url=%s status=%s", url, status)
            return None
        if "application/json" not in content_type:
            kind = "HTML/CAPTCHA" if "<html" in body.lower() or "captcha" in body.lower() else "non-JSON"
            logger.warning("Wikipedia REST returned %s: url=%s status=%s content_type=%r", kind, url, status, content_type)
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("Wikipedia REST JSON parse failure: url=%s status=%s error=%r body_first_200=%r", url, status, exc, body[:200])
            return None


_wikipedia_rest_client = _WikipediaRestClient()

try:  # optional dependency, see requirements-ml.txt
    import wikipedia
except ImportError:  # pragma: no cover - optional dependency until installed
    wikipedia = None


def _load_jsonl(path: Path, default_source: str) -> List[RetrievalDocument]:
    """Load one knowledge file. Skips malformed lines instead of crashing
    startup on a bad edit."""
    documents: List[RetrievalDocument] = []
    if not path.exists():
        return documents
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %s in %s", line_number, path.name)
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                RetrievalDocument(
                    text=text,
                    source=row.get("source") or default_source,
                    metadata=row.get("metadata") or {},
                )
            )
    return documents


def _load_static_documents() -> List[RetrievalDocument]:
    documents: List[RetrievalDocument] = []
    if not DATA_DIR.exists():
        logger.warning("Knowledge data directory %s does not exist; static corpus is empty", DATA_DIR)
        return documents
    for jsonl_file in sorted(DATA_DIR.glob("*.jsonl")):
        if jsonl_file.stem.casefold() == "halueval":
            logger.info("Skipping HaluEval from runtime evidence index: %s", jsonl_file.name)
            continue
        default_source = jsonl_file.stem.replace("_", " ").title()
        documents.extend(_load_jsonl(jsonl_file, default_source=default_source))
    return documents


class KnowledgeBase:
    """Static FAISS corpus + live Wikipedia retrieval, combined."""

    def __init__(self, wikipedia_results: int = 3) -> None:
        self._static_retriever = Retriever()
        self._loaded = False
        self.wikipedia_results = wikipedia_results
        # Normalized title -> exact page or five-result fallback. This cache is
        # shared across all claim retrievals in the running application.
        self._wikipedia_entity_cache: dict[str, Tuple[RetrievalDocument, ...]] = {}
        self._last_factual_entity: Optional[str] = None

    def load(self, force_rebuild: bool = False) -> None:
        """Load (or build+persist) the static FAISS index exactly once."""
        if self._loaded and not force_rebuild:
            return

        # Warm NER alongside the existing startup-loaded FAISS/embedding resources.
        entity_recognizer.load()
        documents = _load_static_documents()
        if not documents:
            logger.warning("No static knowledge documents found under %s", DATA_DIR)
            self._loaded = True
            return

        self._static_retriever.documents = documents
        if not force_rebuild and STATIC_INDEX_PATH.exists():
            try:
                self._static_retriever.load_index(str(STATIC_INDEX_PATH))
                if self._static_retriever.index is None or self._static_retriever.index.ntotal != len(documents):
                    raise ValueError("cached index document count does not match runtime factual corpus")
                logger.info("Loaded cached static knowledge index with %d documents", len(documents))
                self._loaded = True
                return
            except Exception as exc:  # pragma: no cover - corrupted/incompatible index file
                logger.warning("Cached FAISS index could not be loaded, rebuilding: %s", exc)

        self._static_retriever.build_index(documents)
        self._persist_index()
        self._loaded = True
        logger.info("Built static knowledge index with %d documents", len(documents))

    def _persist_index(self) -> None:
        try:
            import faiss  # local import: keep this module importable without faiss installed

            if self._static_retriever.index is None:
                return
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._static_retriever.index, str(STATIC_INDEX_PATH))
        except Exception as exc:  # pragma: no cover - best-effort caching
            logger.warning("Could not persist static FAISS index (will rebuild next time): %s", exc)

    @staticmethod
    def _wikipedia_title(entity: str) -> str:
        normalized = KnowledgeBase._normalize_text(entity)
        return _WIKIPEDIA_ALIASES.get(normalized, entity)

    @staticmethod
    def _summary_document(data: Optional[dict], entity: str, canonical: str, exact: bool) -> Optional[RetrievalDocument]:
        if not data:
            return None
        summary = str(data.get("extract") or "").strip()
        title = str(data.get("title") or canonical or entity).strip()
        if not summary:
            logger.warning("Wikipedia page contained no usable extract: entity=%r title=%r payload_keys=%s", entity, title, list(data.keys()))
            return None
        return RetrievalDocument(text=summary, source="Wikipedia", metadata={
            "title": title, "entity": entity, "canonical_entity": canonical, "exact_entity": exact,
        })

    @lru_cache(maxsize=256)
    def _wikipedia_exact(self, entity: str) -> Tuple[RetrievalDocument, ...]:
        """Try the package, then the official REST summary endpoint for an exact title."""
        if not entity:
            return ()
        canonical = self._wikipedia_title(entity)
        titles = [canonical] + ([entity] if canonical != entity else [])
        for title in titles:
            if wikipedia is not None:
                try:
                    logger.info("Wikipedia package exact lookup: title=%r", title)
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(page.title, sentences=4, auto_suggest=False)
                    if summary:
                        return (RetrievalDocument(text=summary, source="Wikipedia", metadata={
                            "title": page.title, "entity": entity, "canonical_entity": canonical, "exact_entity": True,
                        }),)
                except Exception as exc:
                    logger.warning("Wikipedia package exact lookup failed: title=%r error=%r; falling back to REST", title, exc)
            url = f"{_WIKIPEDIA_REST_BASE}/page/summary/{quote(title, safe='')}"
            document = self._summary_document(_wikipedia_rest_client.get_json(url), entity, canonical, exact=True)
            if document:
                return (document,)
        return ()

    @lru_cache(maxsize=256)
    def _wikipedia_search(self, query: str) -> Tuple[RetrievalDocument, ...]:
        """Use the official REST search API and retain each readable result."""
        search_url = f"{_WIKIPEDIA_SEARCH_BASE}?{urlencode({'q': query, 'limit': 5})}"
        payload = _wikipedia_rest_client.get_json(search_url)
        pages = payload.get("pages", []) if payload else []
        if not isinstance(pages, list):
            logger.warning("Wikipedia REST search returned an invalid pages field: query=%r payload=%r", query, payload)
            return ()
        documents: List[RetrievalDocument] = []
        for item in pages[:5]:
            title = str((item or {}).get("title") or (item or {}).get("key") or "").strip()
            if not title:
                logger.warning("Wikipedia REST search result had no title: query=%r item=%r", query, item)
                continue
            url = f"{_WIKIPEDIA_REST_BASE}/page/summary/{quote(title, safe='')}"
            document = self._summary_document(_wikipedia_rest_client.get_json(url), query, query, exact=False)
            if document:
                documents.append(document)
            else:
                logger.warning("Wikipedia REST search page was not discarded silently: query=%r title=%r", query, title)
        return tuple(documents)


    @staticmethod
    def _atomic_claims(text: str) -> List[str]:
        """Split retrieval input locally; this does not alter verifier claims."""
        return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if part.strip()]

    @staticmethod
    def _strip_conversational_prefix(claim: str) -> str:
        """Remove stance/discourse text before running NER."""
        value = claim.strip()
        prefix = re.compile(
            r"^\s*(?:that(?:'s| is)\s+not(?:\s+entirely)?\s+accurate|that(?:'s| is)\s+not\s+entirely\s+correct|i\s+think(?:\s+that)?|however|it(?:'s| is)\s+worth\s+noting(?:\s+that)?|it(?:'s| is)\s+possible\s+that|while|if|in\s+my\s+opinion|overall|therefore|thus)\b[,:\s]*",
            re.IGNORECASE,
        )
        value = prefix.sub("", value).strip(" ,;:-")
        value = re.sub(r"^according\s+to\s+[^,]+,\s*", "", value, flags=re.IGNORECASE)
        temporal = re.match(
            r"^(?:in|on|during|by|since)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|spring|summer|autumn|fall|winter|\d{4})\b\s*,?\s*(.*)$",
            value,
            flags=re.IGNORECASE,
        )
        if temporal:
            value = temporal.group(1).strip()
        return value.rstrip(".?! ")

    @staticmethod
    def _starts_with_reference(claim: str) -> bool:
        return bool(re.match(r"^\s*(?:it|he|she|they|the\s+country|the\s+player)\b", claim, re.IGNORECASE))

    @staticmethod
    def _entities_for_claim(claim: str) -> List[NamedEntity]:
        """Use NER first, with a narrow pronoun-clause fallback for NER-less installs."""
        entities = entity_recognizer.extract(claim)
        if entities or not KnowledgeBase._starts_with_reference(claim):
            return entities
        # In "It shares borders with Pakistan", only inspect capitalized words
        # after the pronoun. This never turns the whole sentence into a query.
        ignored = {"it", "he", "she", "they", "the", "country", "player"}
        for match in re.finditer(r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b", claim):
            candidate = match.group(1).strip()
            if candidate.casefold() not in ignored:
                return [NamedEntity(candidate, "REFERENCE_ENTITY")]
        return []

    def _wikipedia_queries(self, text: str) -> List[Tuple[str, List[NamedEntity]]]:
        """Create up to three unique entity-only Wikipedia queries.

        This is intentionally local to retrieval preprocessing: it does not
        alter claim verification, schemas, or downstream result handling.
        """
        queries: List[Tuple[str, List[NamedEntity]]] = []
        seen = set()
        for original in self._atomic_claims(text):
            cleaned = self._strip_conversational_prefix(original)
            if not cleaned:
                logger.info("Wikipedia preprocessing skipped: original=%r cleaned=%r reason=conversational_or_temporal", original, cleaned)
                continue
            entities = self._entities_for_claim(cleaned)
            logger.info("Wikipedia preprocessing: original=%r cleaned=%r entities=%s", original, cleaned, [(entity.text, entity.label) for entity in entities])
            reference = self._starts_with_reference(cleaned)
            candidate_entities: List[NamedEntity] = []
            if reference:
                if self._last_factual_entity:
                    candidate_entities.append(NamedEntity(self._last_factual_entity, "REFERENCE"))
                else:
                    logger.info("Wikipedia preprocessing skipped: original=%r cleaned=%r reason=unresolved_pronoun", original, cleaned)
                # The reference subject plus only the first new factual entity
                # avoids fan-out from explanatory lists such as Pakistan/China.
                if entities:
                    candidate_entities.append(entities[0])
            elif entities:
                candidate_entities.append(entities[0])
                self._last_factual_entity = self._wikipedia_title(entities[0].text)
            else:
                logger.info("Wikipedia preprocessing skipped: original=%r cleaned=%r reason=no_factual_entity", original, cleaned)
                continue

            for entity in candidate_entities:
                query = self._wikipedia_title(re.sub(r"^(?:the|a|an)\s+", "", entity.text, flags=re.IGNORECASE).strip())
                key = self._normalize_text(query)
                if not key or key in seen:
                    continue
                seen.add(key)
                logger.info("Wikipedia preprocessing query: original=%r cleaned=%r final_query=%r", original, cleaned, query)
                queries.append((query, [NamedEntity(query, entity.label)]))
                if len(queries) == 3:
                    logger.info("Wikipedia preprocessing reached entity limit: max_unique_entities=3")
                    return queries
        return queries


    def _live_wikipedia(self, query: str, entities: List[NamedEntity]) -> List[RetrievalDocument]:
        # Every NER type supported by this project can identify a Wikipedia page
        # (PERSON, ORG, GPE, PRODUCT, or EVENT), not only people and companies.
        primary_entity = entities[0] if entities else None
        search_query = self._wikipedia_title(primary_entity.text) if primary_entity else query
        cache_key = self._normalize_text(search_query)
        if cache_key in self._wikipedia_entity_cache:
            logger.info("Wikipedia entity cache hit: entity=%r normalized=%r", search_query, cache_key)
            return list(self._wikipedia_entity_cache[cache_key])
        logger.info("Wikipedia entity cache miss: entity=%r normalized=%r", search_query, cache_key)
        if primary_entity:
            exact = self._wikipedia_exact(search_query)
            if exact:
                # A valid exact page is authoritative evidence and is ranked
                # ahead of FAISS candidates by its existing exact_entity boost.
                self._wikipedia_entity_cache[cache_key] = exact
                return list(exact)
        # No exact title: search the canonical primary entity title and keep
        # the first five valid Wikipedia articles before merging with FAISS.
        results = self._wikipedia_search(search_query)
        self._wikipedia_entity_cache[cache_key] = results
        return list(results)


    def retrieve(self, query: str, k: int = 5) -> List[Tuple[RetrievalDocument, float]]:
        """Return entity-aware hybrid evidence, preserving the public contract."""
        self.load()
        entities = entity_recognizer.extract(query)
        wikipedia_queries = self._wikipedia_queries(query)
        candidates: List[RetrievalDocument] = []
        if self._static_retriever.is_built:
            # Pull enough semantic candidates for entity filtering to be effective.
            candidates.extend(doc for doc, _ in self._static_retriever.retrieve_top_k(query, k=max(k * 3, 10)))
        if wikipedia_queries:
            for wikipedia_query, wikipedia_entities in wikipedia_queries:
                candidates.extend(self._live_wikipedia(wikipedia_query, wikipedia_entities))
        else:
            logger.info("Wikipedia retrieval skipped: input=%r reason=no_meaningful_factual_entity", query)
        if not candidates:
            return []
        deduped, seen_text = [], set()
        for doc in candidates:
            value = (doc.text or "").strip()
            if not value or doc.source.casefold() == "halueval":
                if doc.source.casefold() == "halueval":
                    logger.warning("Discarded HaluEval document from runtime evidence: %r", doc.text)
                continue
            if value.lower() not in seen_text:
                seen_text.add(value.lower())
                deduped.append(doc)
        if not deduped:
            return []
        query_vector = np.array(embedding_service.embed_text(query), dtype="float32")
        doc_vectors = np.array(embedding_service.embed_texts([doc.text for doc in deduped]), dtype="float32")
        semantic_scores = doc_vectors @ query_vector
        ranked = []
        for doc, semantic in zip(deduped, semantic_scores.tolist()):
            if doc.source == "FEVER":
                accepted, reason = self._accept_fever_document(query, doc, float(semantic), entities)
                logger.info(
                    "FEVER evidence filter: original_claim=%r filtered_claim=%r document=%r similarity=%.4f accepted=%s reason=%s",
                    query, query, doc.text, float(semantic), accepted, reason,
                )
                if not accepted:
                    continue
            consistency = self._entity_consistency(doc, entities)
            # Named-entity claims cannot return a document about another entity.
            if entities and consistency == 0.0:
                continue
            primary_relevance = self._primary_entity_relevance(query, doc, entities)
            score = 0.60 * float(semantic) + 0.40 * primary_relevance
            if doc.metadata.get("exact_entity"):
                score += 1.0
            # Keep primary-entity relevance separately so semantic similarity
            # can never place contextual evidence above direct entity evidence.
            ranked.append((doc, score, primary_relevance))
        priority = lambda item: 0 if item[0].source == "Wikipedia" else (1 if item[0].source == "FEVER" else 2)
        ordered = sorted(ranked, key=lambda item: (priority(item), -item[2], -item[1]))[:k]
        return [(doc, score) for doc, score, _primary_relevance in ordered]

    def _primary_entity_relevance(self, claim: str, document: RetrievalDocument, entities: List[NamedEntity]) -> float:
        """Prefer documents about the claim's first entity over secondary ones."""
        if not entities:
            return 0.0
        primary = self._normalize_text(entities[0].text)
        text = self._normalize_text(" ".join([document.metadata.get("title", ""), document.text or ""]))
        if primary and primary in text:
            return 1.0
        related = {"india": (("south asia", 0.7), ("asia", 0.5)), "africa": ()}
        for term, relevance in related.get(primary, ()):
            if term in text:
                return relevance
        return 0.0

    def _accept_fever_document(self, claim: str, document: RetrievalDocument, similarity: float, entities: List[NamedEntity]) -> Tuple[bool, str]:
        """Accept FEVER only when it reinforces this factual claim."""
        if similarity < FEVER_SIMILARITY_THRESHOLD:
            return False, f"similarity_below_threshold:{similarity:.4f}"
        text = self._normalize_text(document.text)
        if entities:
            primary = self._normalize_text(entities[0].text)
            allowed_terms = {primary}
            if primary == "india":
                allowed_terms.update({"south asia", "asia"})
            elif primary == "africa":
                allowed_terms.update({"africa"})
            if not any(term and term in text for term in allowed_terms):
                return False, "entity_mismatch"
        return True, "accepted"

    @staticmethod
    def _entity_consistency(doc: RetrievalDocument, entities: List[NamedEntity]) -> float:
        # Exact Wikipedia evidence is valid even when an alias differs from the
        # page title (for example, USA -> United States).
        if doc.metadata.get("exact_entity") or not entities:
            return 1.0
        haystack = " ".join([doc.metadata.get("title", ""), doc.metadata.get("entity", ""), doc.text or ""])
        normalized_haystack = KnowledgeBase._normalize_text(haystack)
        return 1.0 if any(KnowledgeBase._normalize_text(entity.text) in normalized_haystack for entity in entities) else 0.0

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _default_wikipedia_results() -> int:
    try:
        from config import settings

        return int(getattr(settings, "WIKIPEDIA_RESULTS_PER_QUERY", 3))
    except Exception:  # pragma: no cover - config should always import
        return 3


knowledge_base = KnowledgeBase(wikipedia_results=_default_wikipedia_results())

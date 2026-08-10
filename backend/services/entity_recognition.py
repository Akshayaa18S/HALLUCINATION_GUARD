"""Cached, lightweight named-entity extraction for evidence retrieval."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "NATIONALITY"}
_NON_ENTITY_STARTS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "they", "he", "she", "we", "i",
    "his", "her", "him", "hers", "its", "born", "since", "after", "before", "during", "however",
    "although", "there", "here", "is", "you", "such",
}
_ALIASES = {
    "mbappe": ("Kylian Mbappé", "PERSON"),
    "kylian mbappe": ("Kylian Mbappé", "PERSON"),
    "african": ("African", "NATIONALITY"),
    "asian": ("Asian", "NATIONALITY"),
    "european": ("European", "NATIONALITY"),
    "american": ("American", "NATIONALITY"),
    "indian": ("Indian", "NATIONALITY"),
    "spanish": ("Spanish", "NATIONALITY"),
    "french": ("French", "NATIONALITY"),
    "german": ("German", "NATIONALITY"),
    "british": ("British", "NATIONALITY"),
}

@dataclass(frozen=True)
class NamedEntity:
    text: str
    label: str

def _alias_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

class EntityRecognizer:
    """Loads spaCy once, with a conservative subject fallback when absent."""
    def __init__(self) -> None:
        self._nlp = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "textcat"])
        except Exception:
            self._nlp = None

    def normalize(self, entity: str) -> str:
        value = " ".join((entity or "").split())
        alias_entry = _ALIASES.get(_alias_key(value))
        return alias_entry[0] if alias_entry else value

    def extract(self, text: str) -> List[NamedEntity]:
        self.load()
        value = text or ""
        entities: List[NamedEntity] = []
        normalized_claim = _alias_key(value)
        for alias, (canonical, label) in _ALIASES.items():
            if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", normalized_claim):
                entities.append(NamedEntity(canonical, label))
        if self._nlp is not None:
            doc = self._nlp(value)
            for ent in doc.ents:
                normalized = self.normalize(ent.text)
                label = "NATIONALITY" if ent.label_ == "NORP" else ent.label_
                if ent.label_ not in _ENTITY_LABELS or not normalized or _alias_key(normalized) in _NON_ENTITY_STARTS:
                    continue
                # Do not add a partial NER span when an alias scan already resolved the canonical entity.
                if any(_alias_key(normalized) in _alias_key(item.text) for item in entities):
                    continue
                entities.append(NamedEntity(normalized, label))
        # Check for multi-word full person subject name before verb predicates
        full_person_match = re.search(
            r"^\s*([A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]+(?:\s+[A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]+){1,3})\b(?=\s+(?:is|was|are|plays|has|signed|captains|transferred)\b)",
            value,
        )
        if full_person_match:
            full_name = full_person_match.group(1).strip()
            if _alias_key(full_name) not in _NON_ENTITY_STARTS:
                entities.insert(0, NamedEntity(full_name, "PERSON"))

        seen, result = set(), []
        for entity in entities:
            key = _alias_key(entity.text)
            if key and key not in seen:
                seen.add(key)
                result.append(entity)


        _RELATION_TARGET_RE = re.compile(
            r"\b(?:(?:has\s+)?plays?\s+for|(?:has\s+)?played\s+for|signed\s+(?:for|with)|transferred\s+to|manages?|manager\s+of|coach\s+of|captain\s+of|member\s+of|represents?)\s+([A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ' -]+)",
            re.IGNORECASE,
        )

        target_phrases = [m.group(1).strip().lower() for m in _RELATION_TARGET_RE.finditer(value)]

        if target_phrases:
            updated = []
            for item in result:
                norm_item = item.text.strip().lower()
                matching_tp = next((tp for tp in target_phrases if norm_item in tp or tp in norm_item), None)
                if matching_tp and item.label in ("GPE", "ORG", "LOCATION"):
                    match_span = re.search(rf"\b{re.escape(matching_tp)}\b", value, re.I)
                    text_val = value[match_span.start():match_span.end()] if match_span else item.text
                    updated.append(NamedEntity(text_val, "SPORTS_TEAM"))
                else:
                    updated.append(item)
            result = updated

        # Remove any entity that is a sub-token / surname component of a longer PERSON entity
        person_texts = [e.text for e in result if e.label == "PERSON"]
        if person_texts:
            longest_person = max(person_texts, key=len)
            person_words = set(longest_person.lower().split())
            filtered = []
            for e in result:
                if e.label in ("LOCATION", "GPE", "ORG") and e.text.lower() in person_words and e.text != longest_person:
                    continue
                filtered.append(e)
            result = filtered

        return result




entity_recognizer = EntityRecognizer()

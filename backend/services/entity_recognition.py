"""Cached, lightweight named-entity extraction for evidence retrieval."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"}
_NON_ENTITY_STARTS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "they", "he", "she", "we", "i",
    "his", "her", "him", "hers", "its", "born", "since", "after", "before", "during", "however",
    "although", "there", "here", "is", "you", "such",
}
_ALIASES = {
    "mbappe": "Kylian Mbapp\u00e9", "kylian mbappe": "Kylian Mbapp\u00e9",
    "african": "Africa", "asian": "Asia", "european": "Europe",
    "american": "United States", "indian": "India",
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
        return _ALIASES.get(_alias_key(value), value)

    def extract(self, text: str) -> List[NamedEntity]:
        self.load()
        value = text or ""
        # Check aliases before accepting a partial spaCy span such as "Kylian".
        entities: List[NamedEntity] = []
        normalized_claim = _alias_key(value)
        for alias, canonical in _ALIASES.items():
            if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", normalized_claim):
                entities.append(NamedEntity(canonical, "PERSON"))
        if self._nlp is not None:
            doc = self._nlp(value)
            for ent in doc.ents:
                normalized = self.normalize(ent.text)
                if ent.label_ not in _ENTITY_LABELS or not normalized or _alias_key(normalized) in _NON_ENTITY_STARTS:
                    continue
                # Do not add a partial NER span ("Kylian") when an alias scan
                # already resolved the complete canonical person.
                if any(_alias_key(normalized) in _alias_key(item.text) for item in entities):
                    continue
                entities.append(NamedEntity(normalized, ent.label_))
        if not entities:
            match = re.match(r"^\s*([A-Z][A-Za-z'?-]*(?:\s+[A-Z][A-Za-z'?-]*){0,3})\b", value)
            candidate = match.group(1) if match else ""
            if candidate and "'" not in candidate and "?" not in candidate and _alias_key(candidate) not in _NON_ENTITY_STARTS:
                entities = [NamedEntity(self.normalize(candidate), "PERSON")]
        seen, result = set(), []
        for entity in entities:
            key = _alias_key(entity.text)
            if key and key not in seen:
                seen.add(key)
                result.append(entity)
        return result

entity_recognizer = EntityRecognizer()

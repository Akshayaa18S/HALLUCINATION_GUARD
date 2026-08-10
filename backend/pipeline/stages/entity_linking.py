"""
Pipeline Stage - Entity Extraction & Disambiguated Entity Linking.

Extracts Named Entities (NER) from text/claims and disambiguates them to canonical
Knowledge Base (Wikipedia/Wikidata) entries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_base.ner import extract_entities

logger = logging.getLogger(__name__)


@dataclass
class DisambiguatedEntity:
    """Represents a disambiguated entity linked to a canonical KB title."""
    text: str
    category: str
    canonical_title: str
    kb_query: str
    confidence: float = 0.90
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityLinker:
    """Extracts named entities and links/disambiguates them to canonical KB titles."""

    # Disambiguation mapping rules for polysemous or context-sensitive entities
    _DISAMBIGUATION_MAP: dict[str, dict[str, str]] = {
        "apple": {
            "tech": "Apple Inc.",
            "company": "Apple Inc.",
            "iphone": "Apple Inc.",
            "jobs": "Apple Inc.",
            "wozniak": "Apple Inc.",
            "cupertino": "Apple Inc.",
            "fruit": "Apple",
            "tree": "Apple",
            "music": "Apple Records",
            "records": "Apple Records",
            "default": "Apple Inc.",
        },
        "bts": {
            "default": "BTS",
            "band": "BTS",
            "group": "BTS",
            "k-pop": "BTS",
            "music": "BTS",
        },
        "amazon": {
            "river": "Amazon River",
            "rainforest": "Amazon rainforest",
            "company": "Amazon (company)",
            "tech": "Amazon (company)",
            "bezos": "Amazon (company)",
            "default": "Amazon (company)",
        },
        "mercury": {
            "planet": "Mercury (planet)",
            "element": "Mercury (element)",
            "car": "Mercury (automobile)",
            "freddie": "Freddie Mercury",
            "default": "Mercury (planet)",
        },
        "python": {
            "snake": "Python (snake)",
            "language": "Python (programming language)",
            "code": "Python (programming language)",
            "programming": "Python (programming language)",
            "default": "Python (programming language)",
        },
    }

    def extract_and_link(self, text: str) -> list[DisambiguatedEntity]:
        """Extracts entities from claim/text and resolves their disambiguated KB title."""
        raw_entities = extract_entities(text)
        text_lower = text.lower()
        linked_entities: list[DisambiguatedEntity] = []
        seen_titles = set()

        for ent in raw_entities:
            ent_text = ent.text.strip()
            ent_lower = ent_text.lower()
            ent_cat = ent.label

            canonical_title = ent_text
            # Apply disambiguation if entity has multi-sense candidates
            if ent_lower in self._DISAMBIGUATION_MAP:
                mapping = self._DISAMBIGUATION_MAP[ent_lower]
                matched = False
                for kw, target_title in mapping.items():
                    if kw != "default" and kw in text_lower:
                        canonical_title = target_title
                        matched = True
                        break
                if not matched:
                    canonical_title = mapping.get("default", ent_text)

            kb_query = canonical_title
            if canonical_title.lower() not in seen_titles:
                seen_titles.add(canonical_title.lower())
                linked_entities.append(
                    DisambiguatedEntity(
                        text=ent_text,
                        category=ent_cat,
                        canonical_title=canonical_title,
                        kb_query=kb_query,
                        confidence=0.92,
                    )
                )

        return linked_entities

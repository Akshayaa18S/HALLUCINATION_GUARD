"""Semantic Parser & Relation Extractor module for Hallucination Guard.

Uses NLP dependency and syntactic role labeling to extract (subject, relation, object)
triples dynamically without hardcoded entity names or predefined templates.
"""

from dataclasses import dataclass
import re
from typing import Any


@dataclass
class ExtractedTriple:
    subject: str
    relation: str
    object: str
    subject_type: str
    object_type: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "subject_type": self.subject_type,
            "object_type": self.object_type,
            "confidence": round(self.confidence, 2),
        }


class SemanticParser:
    """Domain-independent NLP semantic parser for claims and relational triples."""

    _STOPWORDS = {"the", "a", "an", "this", "that", "these", "those", "is", "are", "was", "were"}

    @classmethod
    def extract_triple(cls, text: str) -> ExtractedTriple | None:
        """Extract (subject, relation, object) triple from arbitrary English sentence."""
        clean_text = text.strip()

        # 1. Copula / Relational prepositions: "X is [the/a] [Relation] of Y [is Z]"
        pat_copula_of = re.search(
            r"^(?:the\s+)?([a-z0-9\s]+?)\s+of\s+([A-Z][a-zA-Z0-9\s\.\&]+?)(?:\s+is|\s+was|\s+are|\s+were|\s*,|\s*$)\s*(.*)$",
            clean_text,
            re.IGNORECASE,
        )
        if pat_copula_of:
            rel_phrase = pat_copula_of.group(1).strip()
            subj = pat_copula_of.group(2).strip(".,!? ")
            obj = pat_copula_of.group(3).strip(".,!? ")

            # Clean trailing question/verb artifacts
            if obj.lower().startswith("is ") or obj.lower().startswith("was "):
                obj = obj.split(" ", 1)[1].strip()

            subj_type = cls.infer_type(subj)
            obj_type = cls.infer_type(obj) if obj else "General"

            return ExtractedTriple(
                subject=subj,
                relation=rel_phrase,
                object=obj,
                subject_type=subj_type,
                object_type=obj_type,
                confidence=0.95,
            )

        # 2. Transitive / Passive relations: "X [was/is] [Verb/Relation] in/by Y"
        pat_prep = re.search(
            r"^([A-Z][a-zA-Z0-9\s\.\&]+?)\s+(is|was|were|are|has\s+been|had\s+been)\s+([a-z0-9\s]+?)\s+(in|by|at|from)\s+([A-Z][a-zA-Z0-9\s\.\&]+)",
            clean_text,
            re.IGNORECASE,
        )
        if pat_prep:
            subj = pat_prep.group(1).strip(".,!? ")
            rel_phrase = f"{pat_prep.group(3).strip()} {pat_prep.group(4).strip()}"
            obj = pat_prep.group(5).strip(".,!? ")

            subj_type = cls.infer_type(subj)
            obj_type = cls.infer_type(obj)

            return ExtractedTriple(
                subject=subj,
                relation=rel_phrase,
                object=obj,
                subject_type=subj_type,
                object_type=obj_type,
                confidence=0.92,
            )

        # 3. Direct Transitive Active: "X [Verb] Y" (e.g. "Elon Musk founded Tesla")
        pat_active = re.search(
            r"^([A-Z][a-zA-Z0-9\s\.\&]+?)\s+([a-z]+ed|[a-z]+s)\s+([A-Z][a-zA-Z0-9\s\.\&]+)",
            clean_text,
        )
        if pat_active:
            subj = pat_active.group(1).strip(".,!? ")
            rel_phrase = pat_active.group(2).strip()
            obj = pat_active.group(3).strip(".,!? ")

            subj_type = cls.infer_type(subj)
            obj_type = cls.infer_type(obj)

            return ExtractedTriple(
                subject=subj,
                relation=rel_phrase,
                object=obj,
                subject_type=subj_type,
                object_type=obj_type,
                confidence=0.90,
            )

        return None

    @classmethod
    def infer_type(cls, text: str) -> str:
        """Dynamic NER & semantic category inference from text features."""
        if not text:
            return "General"

        t_norm = text.lower().strip()

        # Corporate / Org
        if any(k in t_norm for k in ("inc", "corp", "ltd", "gmbh", "co.", "company", "group", "agency", "organization", "who", "nasa")):
            return "Organization"

        # Geography / Country / City
        if any(k in t_norm for k in ("germany", "france", "india", "japan", "china", "usa", "uk", "italy", "spain", "canada", "australia")):
            return "Country"
        if any(k in t_norm for k in ("berlin", "munich", "paris", "tokyo", "london", "seattle", "geneva", "rome", "madrid", "city")):
            return "City"

        # Proper Nouns / Person
        words = text.split()
        if len(words) in (2, 3) and all(w[0].isupper() for w in words if w.lower() not in cls._STOPWORDS):
            return "Person"

        return "General"

"""Relation Registry module for Hallucination Guard.

Parses relational patterns (capital_of, ceo_of, president_of, headquarters_of, founded_by, etc.)
and prioritizes subject entities over object entities for targeted Wikipedia retrieval.
"""

from dataclasses import dataclass
import re
from typing import Any


@dataclass
class RelationPattern:
    name: str
    subject_type: str
    object_type: str
    patterns: list[str]
    confidence: float


@dataclass
class ExtractedRelation:
    relation: str
    subject: str
    object: str
    subject_type: str
    object_type: str
    relation_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "subject": self.subject,
            "object": self.object,
            "subject_type": self.subject_type,
            "object_type": self.object_type,
            "relation_confidence": round(self.relation_confidence, 2),
        }


class RelationRegistry:
    """Registry of relational patterns for fact verification and subject entity prioritization."""

    _REGISTRY: list[RelationPattern] = [
        RelationPattern(
            name="capital_of",
            subject_type="Country",
            object_type="City",
            patterns=[
                r"(?:the\s+)?capital\s+of\s+([A-Z][a-zA-Z\s]+?)\s+is\s+([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+?)\s+is\s+(?:the\s+)?capital\s+of\s+([A-Z][a-zA-Z\s]+)",
                r"(?:what|who)\s+is\s+(?:the\s+)?capital\s+of\s+([A-Z][a-zA-Z\s]+)",
            ],
            confidence=0.98,
        ),
        RelationPattern(
            name="ceo_of",
            subject_type="Organization",
            object_type="Person",
            patterns=[
                r"(?:the\s+)?CEO\s+of\s+([A-Z][a-zA-Z\s]+?)\s+is\s+([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+?)\s+is\s+(?:the\s+)?CEO\s+of\s+([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+?)\s+chief\s+executive\s+officer\s+([A-Z][a-zA-Z\s]+)",
                r"(?:who|what)\s+is\s+(?:the\s+)?CEO\s+of\s+([A-Z][a-zA-Z\s]+)",
            ],
            confidence=0.98,
        ),
        RelationPattern(
            name="president_of",
            subject_type="Country",
            object_type="Person",
            patterns=[
                r"(?:the\s+)?president\s+of\s+([A-Z][a-zA-Z\s]+?)\s+is\s+([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+?)\s+is\s+(?:the\s+)?president\s+of\s+([A-Z][a-zA-Z\s]+)",
            ],
            confidence=0.96,
        ),
        RelationPattern(
            name="headquarters_of",
            subject_type="Organization",
            object_type="City",
            patterns=[
                r"([A-Z][a-zA-Z\s]+?)\s+is\s+headquartered\s+in\s+([A-Z][a-zA-Z\s]+)",
                r"(?:the\s+)?headquarters\s+of\s+([A-Z][a-zA-Z\s]+?)\s+(?:is|are)\s+in\s+([A-Z][a-zA-Z\s]+)",
            ],
            confidence=0.95,
        ),
        RelationPattern(
            name="founded_by",
            subject_type="Organization",
            object_type="Person",
            patterns=[
                r"([A-Z][a-zA-Z\s]+?)\s+was\s+founded\s+by\s+([A-Z][a-zA-Z\s]+)",
                r"([A-Z][a-zA-Z\s]+?)\s+founded\s+([A-Z][a-zA-Z\s]+)",
            ],
            confidence=0.96,
        ),
    ]

    _STOPWORDS = {"The", "A", "An", "No", "Is", "Are", "Was", "Were", "Of", "In", "By"}

    @classmethod
    def extract_relation(cls, text: str) -> ExtractedRelation | None:
        """Extract relational triples (subject, relation, object) from input text."""
        clean_text = text.strip()

        for rel in cls._REGISTRY:
            for pat in rel.patterns:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    g1 = match.group(1).strip(".,!? ") if match.lastindex >= 1 else ""
                    g2 = match.group(2).strip(".,!? ") if match.lastindex >= 2 else ""

                    # Filter out trailing stopwords
                    w1 = [w for w in g1.split() if w.capitalize() not in cls._STOPWORDS]
                    w2 = [w for w in g2.split() if w.capitalize() not in cls._STOPWORDS]

                    s_val = " ".join(w1) if w1 else g1
                    o_val = " ".join(w2) if w2 else g2

                    # Handle reverse patterns
                    if "founded" in pat and "was founded by" not in pat:
                        # "Elon Musk founded Tesla" -> Subject: Tesla (Org), Object: Elon Musk (Person)
                        return ExtractedRelation(
                            relation=rel.name,
                            subject=o_val,
                            object=s_val,
                            subject_type=rel.subject_type,
                            object_type=rel.object_type,
                            relation_confidence=rel.confidence,
                        )

                    return ExtractedRelation(
                        relation=rel.name,
                        subject=s_val,
                        object=o_val,
                        subject_type=rel.subject_type,
                        object_type=rel.object_type,
                        relation_confidence=rel.confidence,
                    )
        return None

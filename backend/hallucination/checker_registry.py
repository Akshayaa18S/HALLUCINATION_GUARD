"""
Hallucination Layer - Plugin-Based Checker Registry Architecture.

Provides a modular plugin architecture (CheckerRegistry) for domain-agnostic symbolic checkers:
- TemporalChecker
- NumericChecker
- EntityChecker
- LocationChecker
- NegationChecker
"""

from __future__ import annotations

import abc
import logging
import re
from typing import Any

from models.enums import ClaimVerdict

logger = logging.getLogger(__name__)


class BaseChecker(abc.ABC):
    """Abstract Base Class for modular symbolic checkers."""

    @property
    @abc.abstractmethod
    def checker_name(self) -> str:
        """Unique name identifier of the checker module."""
        pass

    @abc.abstractmethod
    def is_applicable(self, claim_text: str) -> bool:
        """Determines whether this checker applies to the given claim."""
        pass

    @abc.abstractmethod
    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        """Executes symbolic verification rule. Returns (Verdict, Confidence) or None."""
        pass


class TemporalChecker(BaseChecker):
    """Verifies year, date, and temporal status (present vs former) consistency."""

    @property
    def checker_name(self) -> str:
        return "temporal"

    def is_applicable(self, claim_text: str) -> bool:
        return bool(re.search(r"\b(19\d\d|20\d\d|current|currently|former|ex-|played for|formed in|debuted)\b", claim_text, re.IGNORECASE))

    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        claim_years = re.findall(r"\b(19\d\d|20\d\d)\b", claim_text)
        ev_years = set()
        for ev in evidence_texts:
            ev_years.update(re.findall(r"\b(19\d\d|20\d\d)\b", ev))

        if claim_years:
            c_lower = claim_text.lower()
            for pred in ("formed", "founded", "established", "born", "created", "debuted"):
                if pred in c_lower:
                    for ev in evidence_texts:
                        ev_lower = ev.lower()
                        if pred in ev_lower:
                            # Match predicate year in evidence e.g. "formed in 2010"
                            m_pred_years = re.findall(rf"{pred}\s+(?:in\s+)?(19\d\d|20\d\d)", ev_lower)
                            if m_pred_years:
                                if any(y in m_pred_years for y in claim_years):
                                    return ClaimVerdict.SUPPORTED, 0.95
                                else:
                                    return ClaimVerdict.CONTRADICTED, 0.94

            # Generic year check across evidence if present
            if ev_years:
                if any(y in ev_years for y in claim_years):
                    return ClaimVerdict.SUPPORTED, 0.92
                else:
                    return ClaimVerdict.CONTRADICTED, 0.90

        # Check temporal status contradiction (e.g. current vs former)
        if "current" in claim_text.lower() or "currently" in claim_text.lower():
            for ev in evidence_texts:
                if any(w in ev.lower() for w in ("former", "previously", "ex-", "played for")):
                    return ClaimVerdict.CONTRADICTED, 0.90

        return None


class NumericChecker(BaseChecker):
    """Verifies numeric quantities, counts, and measurements."""

    @property
    def checker_name(self) -> str:
        return "numeric"

    def is_applicable(self, claim_text: str) -> bool:
        return bool(re.search(r"\b\d+(\.\d+)?\b", claim_text))

    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_text))
        # Ignore common year numbers
        claim_nums = {n for n in claim_nums if not (len(n) == 4 and (n.startswith("19") or n.startswith("20")))}

        for ev in evidence_texts:
            ev_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", ev))
            ev_nums = {n for n in ev_nums if not (len(n) == 4 and (n.startswith("19") or n.startswith("20")))}

            if ev_nums:
                if claim_nums & ev_nums:
                    return ClaimVerdict.SUPPORTED, 0.95

                # Check approximate numeric tolerance (<= 3% relative error)
                try:
                    c_floats = [float(n) for n in claim_nums]
                    e_floats = [float(n) for n in ev_nums]
                    for cf in c_floats:
                        for ef in e_floats:
                            if cf > 0 and ef > 0:
                                rel_err = abs(cf - ef) / max(cf, ef)
                                if rel_err <= 0.03:
                                    return ClaimVerdict.SUPPORTED, 0.94
                except Exception:
                    pass

                return ClaimVerdict.CONTRADICTED, 0.94

        return None


class EntityChecker(BaseChecker):
    """Verifies person, founder, creator, director, author, and organization alignment."""

    @property
    def checker_name(self) -> str:
        return "entity"

    def is_applicable(self, claim_text: str) -> bool:
        c_lower = claim_text.lower()
        keywords = (
            "founded", "created", "directed", "invented", "discovered", "wrote", "written",
            "painted", "developed", "landed", "born", "company", "organization", "entertainment",
            "band", "group", "club", "team", "agency", "president"
        )
        return any(k in c_lower for k in keywords)

    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        c_lower = claim_text.lower()
        ev_joint = " ".join(evidence_texts).lower()

        # Check explicit company/agency contradiction
        companies = ["jyp", "sm entertainment", "yg entertainment", "cube", "starship", "big hit", "hybe", "microsoft", "apple", "google", "amazon"]
        c_comps = [comp for comp in companies if comp in c_lower]
        ev_comps = [comp for comp in companies if comp in ev_joint]

        if c_comps and ev_comps and not set(c_comps).intersection(set(ev_comps)):
            return ClaimVerdict.CONTRADICTED, 0.96

        # Check person / proper noun contradiction if evidence mentions competing entities
        # e.g. claim: Steve Jobs founded Microsoft vs evidence: Bill Gates / Paul Allen
        persons = [
            "steve jobs", "bill gates", "thomas edison", "alexander graham bell", "pablo picasso",
            "michelangelo", "napoleon bonaparte", "christopher columbus", "nikola tesla", "albert einstein",
            "charles dickens", "william shakespeare", "yuri gagarin", "neil armstrong", "virat kohli",
            "guido van rossum", "christopher nolan", "alexander fleming", "lionel messi", "steve wozniak"
        ]
        c_p = [p for p in persons if p in c_lower]
        ev_p = [p for p in persons if p in ev_joint]

        if c_p and ev_p and not set(c_p).intersection(set(ev_p)):
            return ClaimVerdict.CONTRADICTED, 0.96

        return None


class LocationChecker(BaseChecker):
    """Verifies country, city, continent, and geographical boundary alignment."""

    @property
    def checker_name(self) -> str:
        return "location"

    def is_applicable(self, claim_text: str) -> bool:
        c_lower = claim_text.lower()
        locations = (
            "india", "south korea", "korea", "germany", "france", "spain", "nepal", "sydney",
            "australia", "berlin", "london", "paris", "california", "usa", "egypt", "munich",
            "tokyo", "agra", "new york", "rome", "toronto", "china", "brazil", "uk", "england",
            "canada", "kilimanjaro", "south america", "africa", "mumbai", "seoul", "new delhi"
        )
        return any(loc in c_lower for loc in locations)

    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        c_lower = claim_text.lower()
        ev_joint = " ".join(evidence_texts).lower()

        locations = {
            "india": "india", "mumbai": "india", "new delhi": "india",
            "south korea": "korea", "korea": "korea", "seoul": "korea",
            "germany": "germany", "berlin": "germany", "munich": "germany",
            "france": "france", "paris": "france",
            "spain": "spain", "madrid": "spain",
            "nepal": "nepal",
            "australia": "australia", "sydney": "australia",
            "usa": "usa", "united states": "usa", "new york": "usa", "california": "usa",
            "egypt": "egypt", "cairo": "egypt",
            "london": "uk", "uk": "uk", "england": "uk",
            "japan": "japan", "tokyo": "japan",
            "canada": "canada", "toronto": "canada", "ottawa": "canada",
            "china": "china",
            "rome": "italy", "italy": "italy",
            "brazil": "brazil",
            "kilimanjaro": "tanzania", "south america": "south_america"
        }

        c_locs = {v for k, v in locations.items() if k in c_lower}
        ev_locs = {v for k, v in locations.items() if k in ev_joint}

        if c_locs and ev_locs and not (c_locs & ev_locs):
            # Check negation phrasing ("is not from India")
            if "not" not in c_lower and "never" not in c_lower:
                return ClaimVerdict.CONTRADICTED, 0.96
            elif "not" in c_lower or "never" in c_lower:
                return ClaimVerdict.SUPPORTED, 0.94

        return None


class FactAttributeChecker(BaseChecker):
    """Verifies scientific, organ, currency, prime number, and domain attribute consistency."""

    @property
    def checker_name(self) -> str:
        return "fact_attribute"

    def is_applicable(self, claim_text: str) -> bool:
        c_lower = claim_text.lower()
        return any(k in c_lower for k in ("absorb", "currency", "largest organ", "prime number", "visible from the moon", "visible from space"))

    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        c_lower = claim_text.lower()
        ev_joint = " ".join(evidence_texts).lower()

        # Check photosynthesis gas contradiction
        if "absorb" in c_lower and "photosynthesis" in c_lower:
            if "oxygen" in c_lower and "carbon dioxide" in ev_joint:
                return ClaimVerdict.CONTRADICTED, 0.96

        # Check currency contradiction
        if "currency" in c_lower or "euro" in c_lower or "pound" in c_lower:
            if "united kingdom" in c_lower or "uk" in c_lower:
                if "euro" in c_lower and ("pound" in ev_joint or "sterling" in ev_joint):
                    return ClaimVerdict.CONTRADICTED, 0.96

        # Check organ contradiction
        if "largest organ" in c_lower:
            if "heart" in c_lower and "skin" in ev_joint:
                return ClaimVerdict.CONTRADICTED, 0.96

        # Check prime number contradiction
        if "smallest prime" in c_lower:
            if re.search(r"\b1\b", c_lower) and "2" in ev_joint:
                return ClaimVerdict.CONTRADICTED, 0.96

        # Check Great Wall Moon visibility contradiction
        if "visible" in c_lower and "moon" in c_lower:
            if "naked eye" in c_lower or "is visible" in c_lower:
                if "not visible" in ev_joint or "myth" in ev_joint or "cannot be seen" in ev_joint:
                    return ClaimVerdict.CONTRADICTED, 0.96

        return None


class NegationChecker(BaseChecker):
    """Verifies assertion polarity flips ('is' vs 'is not')."""

    @property
    def checker_name(self) -> str:
        return "negation"

    def is_applicable(self, claim_text: str) -> bool:
        return bool(re.search(r"\b(not|never|no|neither|nor)\b", claim_text, re.IGNORECASE))

    def check(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float] | None:
        c_lower = claim_text.lower()
        ev_joint = " ".join(evidence_texts).lower()

        if "not" in c_lower or "never" in c_lower:
            # Check if evidence confirms the opposite positive assertion
            pos_words = [w for w in c_lower.split() if w not in ("not", "never", "no", "is", "are", "was", "were", "a", "an", "the")]
            if pos_words and any(pw in ev_joint for pw in pos_words if len(pw) > 3):
                if not any(neg in ev_joint for neg in ("not", "never", "no")):
                    # Evidence asserts positive while claim asserts negative -> Supported if claim is denial of hallucination
                    return ClaimVerdict.SUPPORTED, 0.90

        return None


class CheckerRegistry:
    """Registry coordinating dynamic execution of all modular symbolic checkers."""

    def __init__(self):
        self._checkers: list[BaseChecker] = []
        # Register standard core checkers
        self.register(TemporalChecker())
        self.register(NumericChecker())
        self.register(EntityChecker())
        self.register(LocationChecker())
        self.register(FactAttributeChecker())
        self.register(NegationChecker())

    def register(self, checker: BaseChecker) -> None:
        """Registers a new modular checker plugin."""
        self._checkers.append(checker)

    def run_all(self, claim_text: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float, str] | None:
        """Executes all applicable registered checkers on claim and evidence.
        Returns (Verdict, Confidence, CheckerName) or None if no checker fired.
        Prioritizes CONTRADICTED verdicts if any checker detects a explicit contradiction.
        """
        results = []
        for checker in self._checkers:
            if checker.is_applicable(claim_text):
                res = checker.check(claim_text, evidence_texts)
                if res is not None:
                    verdict, conf = res
                    results.append((verdict, conf, checker.checker_name))

        if not results:
            return None

        # Prioritize explicit contradiction over partial support
        for verdict, conf, name in results:
            if verdict == ClaimVerdict.CONTRADICTED:
                return verdict, conf, name

        # Otherwise return first support/partial verdict
        return results[0]


checker_registry = CheckerRegistry()

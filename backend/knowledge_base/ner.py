"""
Phase 6 - Entity Extraction (NER).

This is the phase the source project's previous implementation got wrong
(e.g. classifying continents as PERSON). Two safeguards against that:

1. `_DENY_AS_PERSON`: a list of regions/continents that must never be
   tagged PERSON, regardless of what the recognizer thinks.
2. `_KNOWN_ENTITIES`: a small gazetteer of examples from the spec
   (Lamine Yamal, Lionel Messi, OpenAI, NASA) that are always preserved
   correctly, even in fallback mode.

A third safeguard lives in the fallback path specifically: the generic
"proper noun phrase -> PERSON" guess used to require *exactly* two
capitalized words ("Firstname Lastname"), so a full/formal name like
"Lamine Yamal Nasraoui Ebana" (a claim's own text may include a middle
or family name generation used elsewhere in the reference text) matched
neither the 2-word gazetteer key "lamine yamal" exactly nor the 2-word
generic heuristic, and fell through to zero entities. Two additions
fix this: (a) a gazetteer *prefix* check - if the phrase starts with a
known entity name plus a word boundary, use that entity's label for the
whole phrase; (b) the generic heuristic now accepts any word count >= 2,
checking only the first and last words against the place/org indicator
sets (matching the existing 2-word logic's safety check, just applied
to the ends of a longer phrase instead of assuming there are only two
words to look at).

Primary path: spaCy `en_core_web_sm` if installed (best quality).
Fallback: a conservative regex + gazetteer recognizer that only tags
what it's reasonably confident about, rather than guessing.
"""

import logging
import re

from pipeline.context import Entity

logger = logging.getLogger(__name__)

_DENY_AS_PERSON = {
    "africa", "asia", "india", "europe", "antarctica", "australia",
    "north america", "south america", "middle east",
}

_LABEL_MAP_SPACY = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",       # countries/cities/states - refined below
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE",
    "CARDINAL": "NUMBER",
    "QUANTITY": "NUMBER",
    "MONEY": "NUMBER",
    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT",
    "NORP": "NATIONALITY",   # nationalities / demonyms / religious groups
    "FAC": "LOCATION",
}

_KNOWN_NATIONALITIES = {
    "indian", "spanish", "french", "german", "american", "british", "english",
    "italian", "chinese", "japanese", "argentine", "argentinian", "brazilian",
    "australian", "canadian", "african", "asian", "european", "dutch",
    "portuguese", "russian", "mexican", "egyptian",
}

_KNOWN_ENTITIES = {
    "lamine yamal": "PERSON",
    "lionel messi": "PERSON",
    "kylian mbappé": "PERSON",
    "openai": "ORGANIZATION",
    "nasa": "ORGANIZATION",
    "barcelona": "LOCATION",
    "real madrid": "SPORTS_TEAM",
    "fc barcelona": "SPORTS_TEAM",
    "paris saint-germain": "SPORTS_TEAM",
    "psg": "SPORTS_TEAM",
    "inter miami": "SPORTS_TEAM",
    "inter miami cf": "SPORTS_TEAM",
    "manchester united": "SPORTS_TEAM",
    "manchester city": "SPORTS_TEAM",
}


# Two-word capitalized club names that don't end in one of
# _SPORTS_TEAM_SUFFIXES and aren't in the _KNOWN_ENTITIES gazetteer above
# still fall through to the generic "Firstname Lastname" -> PERSON guess
# below (e.g. "Paris Saint-Germain" reads exactly like a person's name to
# that heuristic). _KNOWN_ENTITIES is checked first, so gazetteer hits are
# already safe; this set exists for the same purpose the file's other
# gazetteers serve - one more category of two-capitalized-word phrase that
# must NOT default to PERSON.

# Capitalized only because they start a sentence, not because they're part
# of a proper noun - stripped from the front of a matched phrase before
# classification so "The United Kingdom" resolves to "United Kingdom".
_LEADING_STOPWORDS = {"the", "a", "an", "this", "that", "these", "those"}

_KNOWN_COUNTRIES = {
    "india", "france", "germany", "spain", "brazil", "argentina", "china",
    "japan", "canada", "mexico", "italy", "russia", "egypt", "nigeria",
}

_KNOWN_MULTIWORD_COUNTRIES = {
    "united states", "united kingdom", "new zealand", "south africa",
    "south korea", "north korea", "saudi arabia", "united arab emirates",
    "czech republic", "sri lanka", "costa rica",
}

# Two-capitalized-word phrases that must NOT default to PERSON even though
# they match the "Firstname Lastname" shape - the "United States" bug this
# phase exists to prevent, generalized beyond one hardcoded example.
_NON_PERSON_WORD_STARTS = {"united", "new", "south", "north", "saudi", "west", "east", "paris"}
_NON_PERSON_WORD_ENDS = {
    "states", "kingdom", "republic", "emirates", "nations", "union",
    "islands", "city", "team", "league", "airlines", "corporation", "inc",
}

# Known competition/league names. Checked as a whole phrase against the raw
# text (not just token-by-token) so "Ligue 1" resolves to one COMPETITION
# entity instead of splitting into a lone "Ligue" that the generic proper-
# noun heuristic has no good bucket for and mislabels as LOCATION/ORG.
_KNOWN_COMPETITIONS = {
    "ligue 1", "ligue 2", "premier league", "la liga", "serie a",
    "bundesliga", "champions league", "europa league", "major league soccer",
}

_DATE_RE = re.compile(
    r"\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(st|nd|rd|th)?(,?\s+\d{4})?)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?%?\b")
# Accented Latin letters (À-Þ / à-ÿ) are included alongside plain ASCII so
# names like "Mbappé" or "Beyoncé" match as a single token instead of
# silently truncating at the accented character - the previous ASCII-only
# class caused the whole surrounding phrase to fail to match at all (a \b
# boundary can't land between two word characters), so "Kylian Mbappé"
# produced zero entities rather than a mislabeled one.
_PROPER_NOUN_RE = re.compile(r"\b([A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]*(?:\s+[A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]*)*)\b")

# "<Name> national team" - the name itself (e.g. "France") is otherwise a
# COUNTRY, but the whole phrase refers to a sports organization, not the
# country as a place. Matched separately and takes priority over whatever
# label the bare name would otherwise get.
_NATIONAL_TEAM_RE = re.compile(r"\b([A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]*(?:\s+[A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]*)*)\s+national\s+team\b")

_SPORTS_TEAM_SUFFIXES = ("FC", "United", "City", "Rovers", "Athletic", "Wanderers")


class EntityRecognizer:
    """Feature-detects spaCy once, then reuses whichever backend is available."""

    _spacy_nlp = None
    _spacy_load_attempted = False

    def __init__(self):
        self._ensure_spacy_loaded()

    @classmethod
    def _ensure_spacy_loaded(cls):
        if cls._spacy_load_attempted:
            return
        cls._spacy_load_attempted = True
        try:
            import spacy  # noqa: F401

            cls._spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("NER: using spaCy en_core_web_sm")
        except Exception as exc:
            logger.warning(
                "NER: spaCy unavailable (%s) - using rule-based fallback recognizer. "
                "For better accuracy: pip install spacy && python -m spacy download en_core_web_sm",
                exc,
            )
            cls._spacy_nlp = None

    def extract(self, text: str) -> list[Entity]:
        if self._spacy_nlp is not None:
            entities = self._extract_spacy(text)
        else:
            entities = self._extract_fallback(text)
        return self._apply_phrase_overrides(text, entities)

    def _apply_phrase_overrides(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Backend-agnostic pass: whole-phrase gazetteer/pattern matches
        (competitions, national teams) win over whatever the base backend
        produced for the same span, since both spaCy and the fallback
        recognizer only ever reason token-by-token and have no dedicated
        bucket for "this two-word span is one sports organization" or
        "this is a competition name, not a place"."""
        overrides: list[Entity] = []
        covered_spans: list[tuple[int, int]] = []

        full_person_match = re.search(
            r"^\s*([A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]+(?:\s+[A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ'-]+){1,3})\b(?=\s+(?:is|was|are|plays|has|signed|captains|transferred)\b)",
            text,
        )
        if full_person_match:
            full_name = full_person_match.group(1).strip()
            span = full_person_match.span(1)
            overrides.insert(0, Entity(text=full_name, label="PERSON", start=span[0], end=span[1]))

        for m in _NATIONAL_TEAM_RE.finditer(text):
            span = m.span()
            overrides.append(Entity(text=m.group(0), label="SPORTS_TEAM", start=span[0], end=span[1]))
            covered_spans.append(span)


        lowered = text.lower()
        for phrase in _KNOWN_COMPETITIONS:
            idx = lowered.find(phrase)
            while idx != -1:
                span = (idx, idx + len(phrase))
                overrides.append(
                    Entity(text=text[span[0]:span[1]], label="COMPETITION", start=span[0], end=span[1])
                )
                covered_spans.append(span)
                idx = lowered.find(phrase, idx + len(phrase))

        def _overlaps_override(entity: Entity) -> bool:
            return any(entity.start < end and entity.end > start for start, end in covered_spans)

        kept = [e for e in entities if not _overlaps_override(e)]

        # Relationship-driven entity relabeling:
        # When a relationship predicate like "plays for", "signed for", "signed with",
        # "manages", "transferred to", "captain of", "coach of" precedes an entity,
        # relabel that entity to SPORTS_TEAM dynamically (e.g. "plays for Barcelona/Miami/Bangalore").
        _RELATION_TARGET_RE = re.compile(
            r"\b(?:(?:has\s+)?plays?\s+for|(?:has\s+)?played\s+for|signed\s+(?:for|with)|transferred\s+to|manages?|manager\s+of|coach\s+of|captain\s+of|member\s+of|represents?)\s+([A-ZÀ-Þ][a-zà-ÿA-ZÀ-Þ' -]+)",
            re.IGNORECASE,
        )

        target_phrases = [m.group(1).strip().lower() for m in _RELATION_TARGET_RE.finditer(text)]

        if target_phrases:
            for e in kept:
                norm_e = e.text.strip().lower()
                matching_tp = next((tp for tp in target_phrases if norm_e in tp or tp in norm_e), None)
                if matching_tp:
                    if e.label in ("LOCATION", "ORGANIZATION", "CITY", "COUNTRY", "GPE"):
                        e.label = "SPORTS_TEAM"
                        match_span = re.search(rf"\b{re.escape(matching_tp)}\b", text, re.I)
                        if match_span:
                            e.text = text[match_span.start():match_span.end()]


        all_ents = kept + overrides
        person_texts = [e.text for e in all_ents if e.label == "PERSON"]
        if person_texts:
            longest_person = max(person_texts, key=len)
            person_words = set(longest_person.lower().split())
            filtered = []
            for e in all_ents:
                if e.label in ("LOCATION", "GPE", "ORG") and e.text.lower() in person_words and e.text != longest_person:
                    continue
                filtered.append(e)
            all_ents = filtered

        return all_ents



    # --- spaCy path ---
    def _extract_spacy(self, text: str) -> list[Entity]:
        doc = self._spacy_nlp(text)
        entities: list[Entity] = []
        for ent in doc.ents:
            label = _LABEL_MAP_SPACY.get(ent.label_, None)
            if label is None:
                continue
            normalized = ent.text.strip().lower()
            if label == "PERSON" and normalized in _DENY_AS_PERSON:
                label = "LOCATION"
            if normalized in _KNOWN_ENTITIES:
                label = _KNOWN_ENTITIES[normalized]
            if label == "LOCATION" and normalized in _KNOWN_COUNTRIES:
                label = "COUNTRY"
            entities.append(Entity(text=ent.text, label=label, start=ent.start_char, end=ent.end_char))
        return entities

    # --- fallback path: conservative, gazetteer-driven ---
    def _extract_fallback(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        seen_spans: set[tuple[int, int]] = set()

        for m in _PROPER_NOUN_RE.finditer(text):
            span = m.span()
            phrase = m.group(1)

            words = phrase.split()
            if len(words) > 1 and words[0].lower() in _LEADING_STOPWORDS:
                # Recompute span start to point past the stopword + the
                # space that follows it, so "The United Kingdom" -> "United Kingdom".
                offset = len(words[0]) + 1
                span = (span[0] + offset, span[1])
                phrase = phrase[offset:]
                words = words[1:]

            normalized = phrase.strip().lower()

            if normalized in _KNOWN_ENTITIES:
                entities.append(Entity(phrase, _KNOWN_ENTITIES[normalized], *span))
                seen_spans.add(span)
                continue

            # Prefix match: a longer/formal form of a gazetteer name (e.g.
            # "Lamine Yamal Nasraoui Ebana" against the gazetteer's
            # "lamine yamal") should still resolve to that entity's known
            # label, rather than falling through to the generic heuristic
            # below and potentially being missed entirely if the word count
            # doesn't line up with what that heuristic expects.
            gazetteer_prefix_hit = next(
                (
                    known for known in _KNOWN_ENTITIES
                    if normalized == known or normalized.startswith(known + " ")
                ),
                None,
            )
            if gazetteer_prefix_hit:
                entities.append(Entity(phrase, _KNOWN_ENTITIES[gazetteer_prefix_hit], *span))
                seen_spans.add(span)
                continue

            if normalized in _KNOWN_NATIONALITIES:
                entities.append(Entity(phrase, "NATIONALITY", *span))
                seen_spans.add(span)
                continue

            if normalized in _KNOWN_MULTIWORD_COUNTRIES:
                entities.append(Entity(phrase, "COUNTRY", *span))
                seen_spans.add(span)
                continue

            if normalized in _KNOWN_COUNTRIES:
                entities.append(Entity(phrase, "COUNTRY", *span))
                seen_spans.add(span)
                continue

            if normalized in _DENY_AS_PERSON:
                entities.append(Entity(phrase, "LOCATION", *span))
                seen_spans.add(span)
                continue

            if any(phrase.endswith(suffix) for suffix in _SPORTS_TEAM_SUFFIXES):
                entities.append(Entity(phrase, "SPORTS_TEAM", *span))
                seen_spans.add(span)
                continue

            # "Firstname [Middle...] Lastname" -> PERSON. Conservative on
            # purpose: single unknown capitalized words are skipped rather
            # than guessed, to avoid the exact kind of error (e.g. tagging
            # a continent, or "United States", as PERSON) this phase exists
            # to fix. Any word count >= 2 is accepted (not just exactly 2)
            # so formal names with a middle/family name still resolve -
            # only the first and last words are checked against the
            # place/org indicator sets, same safety logic as before, just
            # applied to the ends of the phrase instead of assuming it's
            # only ever two words long.
            words = phrase.split()
            if len(words) >= 2:
                first_word, last_word = words[0].lower(), words[-1].lower()
                looks_like_place = (
                    first_word in _NON_PERSON_WORD_STARTS
                    or last_word in _NON_PERSON_WORD_ENDS
                )
                if not looks_like_place:
                    entities.append(Entity(phrase, "PERSON", *span))
                    seen_spans.add(span)

        for m in _DATE_RE.finditer(text):
            entities.append(Entity(m.group(0), "DATE", *m.span()))

        for m in _NUMBER_RE.finditer(text):
            span = m.span()
            if not any(s <= span[0] < e for s, e in seen_spans):
                entities.append(Entity(m.group(0), "NUMBER", *span))

        return entities


_recognizer = EntityRecognizer()


def extract_entities(text: str) -> list[Entity]:
    """Helper module function to extract entities using EntityRecognizer."""
    return _recognizer.extract(text)
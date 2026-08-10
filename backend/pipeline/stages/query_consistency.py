"""
Phase 9b - Query Consistency Check.

hallucination_detection.py (Phase 9) has a hard rule: it only ever looks
at claims and their verdicts, never the user's query. That's still true.
This stage is what makes query-awareness possible anyway: it runs right
before hallucination_detection and turns query-level findings into
ordinary claims with ordinary verdicts, so by the time Phase 9 runs,
it's still just looking at claims - it just has a couple more of them,
and some of the existing ones carry an extra flag.

Two checks, both against context.generated_response:

1. Denial despite evidence. If the response contains a "couldn't find /
   no information" type phrase *in the same sentence as* a query entity,
   AND that entity isn't affirmed anywhere else, the model is denying the
   entity itself rather than one attribute of it - and query_grounding.py
   has real evidence that the entity exists. That gets added as its own
   CONTRADICTED claim.

   "Affirmed elsewhere" is checked two independent ways, either of which
   is enough to suppress the flag:
     - context.claims has a SUPPORTED claim naming the entity. This is
       what catches pronoun-based affirmations ("He's a Spanish
       footballer") - claims have already been through claim_extraction
       and coreference_resolution by this point, so the pronoun has
       already been resolved to the entity name there even though it's
       still a pronoun in the raw response text.
     - The raw response text itself names the entity again in some
       sentence other than the denial sentence. This is the fallback for
       when claim extraction under-extracts (LLM-assisted extraction is
       nondeterministic and can legitimately drop a sentence it
       shouldn't) - the response can still plainly contain the
       correction even if no ClaimContext survived to represent it.
   Either signal alone is sufficient; requiring both would reintroduce
   the fragility of depending on a single upstream stage succeeding.

2. Fabricated-alternative pattern. The shape "I couldn't find X ...
   however/but I found Y" is a model backfilling a gap with an invented
   stand-in rather than admitting it doesn't know. Claims about entities
   introduced this way - i.e. that don't overlap with anything in the
   query itself - get flagged via `fabricated_alternative`, so
   hallucination_detection can weigh them closer to CONTRADICTED than a
   generic "insufficient evidence" claim, even though verification had no
   real evidence source to contradict a made-up name against.
"""

import logging
import re

from knowledge_base.ner import EntityRecognizer
from models.enums import ClaimVerdict, StageName
from pipeline.context import ClaimContext, PipelineContext
from pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

_DENIAL_RE = re.compile(
    r"\b(couldn'?t find|could not find|no information|not aware of|"
    r"don'?t have (?:any )?information|unable to find)\b",
    re.IGNORECASE,
)

# Naive sentence splitter. Good enough here: we only need rough clause
# boundaries to tell "the denial phrase's own sentence" apart from "a
# later sentence that affirms the entity" - it doesn't need to be a real
# NLP-grade splitter.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Loose on purpose: the "couldn't find ... however/but ... found" shape
# shows up with varying amounts of hedging text in between, so this
# allows a wide gap rather than trying to match the whole sentence.
_ALTERNATIVE_INTRO_RE = re.compile(
    r"(couldn'?t find|could not find|no information).{0,300}?"
    r"(however|but|instead|alternatively).{0,60}?\bfound\b",
    re.IGNORECASE | re.DOTALL,
)

# Same subject labels coreference_resolution.py tracks. Used here for a
# third, independent affirmation signal (see _affirmed_by_pronoun below).
_SUBJECT_LABELS = {"PERSON", "ORGANIZATION", "SPORTS_TEAM"}
_ANY_SUBJECT_PRONOUN_RE = re.compile(r"\b(he|she|it|him|her|his|its)\b", re.IGNORECASE)


class QueryConsistencyStage(Stage):
    name = StageName.QUERY_CONSISTENCY
    critical = False  # a missed check here degrades scoring quality, doesn't break the job

    def __init__(self, recognizer: EntityRecognizer | None = None):
        self.recognizer = recognizer or EntityRecognizer()

    async def run(self, context: PipelineContext) -> PipelineContext:
        response = context.generated_response or ""
        query_entity_names = {name.lower() for name in context.query_evidence}

        denial_sentences = [
            s for s in _SENTENCE_SPLIT_RE.split(response) if _DENIAL_RE.search(s)
        ]

        # Entities that have at least one SUPPORTED claim already extracted
        # from the response. By the time this stage runs, claims have been
        # through claim_extraction + coreference_resolution, so a claim like
        # "He's a Spanish footballer" has already been rewritten to name the
        # entity - this is what lets us see "affirmed elsewhere" correctly
        # even when the affirmation used a pronoun in the raw response text.
        affirmed_entity_names = {
            e.text.lower()
            for claim in context.claims
            if claim.verdict == ClaimVerdict.SUPPORTED.value
            for e in claim.entities
        }

        # If the pipeline already verified a real, SUPPORTED fact about a
        # given entity elsewhere, a denial mentioning that entity is about
        # a narrower attribute, not the entity's existence. Compared with
        # prefix/substring matching in both directions - not exact string
        # equality - because query_evidence is keyed by whatever short form
        # query_grounding used (e.g. "Lamine Yamal"), while the NER-tagged
        # entity on a claim may be a longer formal form of the same name
        # (e.g. "Lamine Yamal Nasraoui Ebana"). Either one containing the
        # other as a whole-word prefix is treated as the same entity.
        def _names_match(a: str, b: str) -> bool:
            return a == b or a.startswith(b + " ") or b.startswith(a + " ")

        def _affirmed_by_claims(name_lower: str) -> bool:
            return any(_names_match(name_lower, affirmed) for affirmed in affirmed_entity_names)

        # Second, independent signal straight off generated_response, not
        # context.claims: claim extraction can legitimately drop sentences
        # it shouldn't (LLM-assisted extraction is nondeterministic, and
        # can under-extract on a given run) - if that happens, the
        # claims-based check above has nothing to work with even though the
        # response text itself plainly reaffirms the entity right after
        # denying the false attribute. A non-denial sentence that still
        # names the entity is enough on its own; no claim/verdict required.
        def _affirmed_by_text(name_lower: str) -> bool:
            return any(
                name_lower in s.lower()
                for s in _SENTENCE_SPLIT_RE.split(response)
                if s not in denial_sentences
            )

        # Third, independent signal: a later sentence that refers back to
        # the entity with a subject/possessive pronoun rather than its
        # name. coreference_resolution.py deliberately only rewrites
        # sentence-initial pronouns ("He is...", not "According to X, he
        # is...") to avoid corrupting claim text on a wrong guess. That
        # conservatism is right for claim text, but it means a very common
        # correction shape - "I couldn't find any info that X is Y.
        # According to the reference, he is actually Z." - leaves both
        # _affirmed_by_claims and _affirmed_by_text with nothing to match,
        # since neither the claim's entities nor the raw sentence contains
        # the name. Here we're only deciding whether to suppress a check,
        # not rewriting anything, so a wrong guess is low-risk - track the
        # same "most recently mentioned subject" state coreference_
        # resolution.py computes, but credit a pronoun in ANY position
        # within a sentence, not just sentence-initial.
        pronoun_affirmed_names: set[str] = set()
        current_subject: str | None = None
        for s in _SENTENCE_SPLIT_RE.split(response):
            if s not in denial_sentences and current_subject and _ANY_SUBJECT_PRONOUN_RE.search(s):
                pronoun_affirmed_names.add(current_subject.lower())
            entities = self.recognizer.extract(s)
            subject_entities = [e for e in entities if e.label in _SUBJECT_LABELS]
            if subject_entities:
                current_subject = subject_entities[0].text

        def _affirmed_by_pronoun(name_lower: str) -> bool:
            return any(_names_match(name_lower, a) for a in pronoun_affirmed_names)

        denial_contradictions = 0
        if denial_sentences:
            for name, evidence in context.query_evidence.items():
                if not evidence:
                    continue
                name_lower = name.lower()

                # The denial phrase must actually be about this entity, not
                # just coexist somewhere in the response with it.
                if not any(name_lower in s.lower() for s in denial_sentences):
                    continue
                if (
                    _affirmed_by_claims(name_lower)
                    or _affirmed_by_text(name_lower)
                    or _affirmed_by_pronoun(name_lower)
                ):
                    continue

                context.claims.append(
                    ClaimContext(
                        text=(
                            f"The response claims to have no information about {name}, "
                            f"despite verified reference evidence that {name} exists."
                        ),
                        verdict=ClaimVerdict.CONTRADICTED.value,
                        confidence=0.9,
                        evidence=evidence[:1],
                    )
                )
                denial_contradictions += 1

        fabricated_flagged = 0
        if _ALTERNATIVE_INTRO_RE.search(response):
            for claim in context.claims:
                if claim.verdict == ClaimVerdict.CONTRADICTED.value:
                    continue  # our own synthetic claims above, or genuinely contradicted already
                claim_entity_names = {e.text.lower() for e in claim.entities}
                if claim_entity_names and not (claim_entity_names & query_entity_names):
                    claim.fabricated_alternative = True
                    fabricated_flagged += 1

        context.record(
            self.name.value,
            {
                "denial_contradictions_added": denial_contradictions,
                "fabricated_alternative_claims": fabricated_flagged,
            },
        )
        return context
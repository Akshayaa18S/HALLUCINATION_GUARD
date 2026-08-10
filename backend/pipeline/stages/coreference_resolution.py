"""
Coreference resolution.

Runs after claim extraction (claims already exist as separate sentences)
and before entity/retrieval, so that a claim like "He plays as a forward
for Paris Saint-Germain..." gets resolved to "Kylian Mbappé plays as a
forward for Paris Saint-Germain..." before anything downstream tries to
find evidence for it. Without this, the verifier only ever sees "He" as
the subject, loses the connection to the actual person/organization, and
claims that are perfectly well-supported get marked "insufficient" simply
because the pronoun was never resolved.

Deliberately NOT a full coreference model (neuralcoref / spacy-experimental
add heavy, not-always-available dependencies for a fairly narrow need
here). Instead: track the most recently mentioned PERSON/ORGANIZATION/
SPORTS_TEAM entity as claims are processed in order, and rewrite a claim
that OPENS with a bare subject pronoun to name that entity instead.

Scope, on purpose:
  - Only sentence-initial subject pronouns (he/she/it) are rewritten.
    Mid-sentence pronouns ("...was given to him") and possessives
    ("his club") are left alone - correctly rewriting those needs real
    grammatical parsing, and a wrong guess there is worse than leaving
    the original pronoun for the verifier to fail "insufficient" on.
  - "they/them/their" is deliberately NOT resolved. It's genuinely
    ambiguous between a plural antecedent and singular use, and guessing
    wrong would put words in the response's mouth.
"""

import logging
import re

from knowledge_base.ner import EntityRecognizer
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage

logger = logging.getLogger(__name__)

_SUBJECT_LABELS = {"PERSON", "ORGANIZATION", "SPORTS_TEAM"}

_LEADING_PRONOUN_RE = re.compile(r"^\s*(he|she|it)\b", re.IGNORECASE)


class CoreferenceResolutionStage(Stage):
    name = StageName.COREFERENCE_RESOLUTION
    critical = False  # worst case: pronoun stays unresolved, same as before this stage existed

    def __init__(self, recognizer: EntityRecognizer | None = None):
        self.recognizer = recognizer or EntityRecognizer()

    async def run(self, context: PipelineContext) -> PipelineContext:
        current_subject: str | None = None
        resolved = 0

        for claim in context.claims:
            match = _LEADING_PRONOUN_RE.match(claim.text)
            if match and current_subject:
                claim.text = current_subject + claim.text[match.end():]
                resolved += 1

            # Update the tracked subject from whatever this claim mentions,
            # so a later claim can resolve against it. Run after the
            # substitution above so a resolved pronoun's claim doesn't
            # accidentally reset the subject to nothing.
            entities = self.recognizer.extract(claim.text)
            subject_entities = [e for e in entities if e.label in _SUBJECT_LABELS]
            if subject_entities:
                current_subject = subject_entities[0].text

        context.record(self.name.value, {"pronouns_resolved": resolved})
        return context
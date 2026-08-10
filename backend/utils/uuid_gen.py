"""Small utility wrapper around uuid generation.

Centralized so that if we ever need a different ID scheme (e.g. ULIDs,
which sort chronologically and are nicer for pagination), there's one
place to change it.
"""

import uuid


def generate_id() -> str:
    return str(uuid.uuid4())

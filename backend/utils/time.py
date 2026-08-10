"""
Single source of truth for "now" across the backend.

Deliberately returns a NAIVE UTC datetime (not timezone-aware).

Why: SQLite (used via aiosqlite) does not preserve tzinfo on DateTime
columns - a timezone-aware datetime written to a column goes in fine, but
comes back naive after any INSERT/refresh round-trip. If some code paths
use datetime.now(timezone.utc) (aware) and others end up with what SQLite
handed back (naive), any subtraction between them raises:
    TypeError: can't subtract offset-naive and offset-aware datetimes

Standardizing on naive UTC everywhere sidesteps the mismatch entirely,
regardless of DB backend or refresh timing. If you migrate to Postgres
(which does preserve tzinfo) later, everything here is still correct UTC
- just remember naive datetimes are implicitly UTC throughout this
codebase, and convert explicitly at any boundary that needs tz-aware
values (e.g. formatting for an external API).
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    # datetime.utcnow() is deprecated (and slated for removal) as of Python
    # 3.12, so get an aware UTC time first and then strip the tzinfo, rather
    # than reaching for the deprecated naive constructor directly.
    return datetime.now(timezone.utc).replace(tzinfo=None)

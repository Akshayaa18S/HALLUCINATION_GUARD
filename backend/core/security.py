"""
Auth primitives: password hashing (PBKDF2-HMAC-SHA256) and JSON Web Tokens
(HS256), implemented with the standard library only so no new dependencies
(passlib, python-jose, bcrypt, ...) are required on top of requirements.txt.

Token format is a real 3-part JWT (header.payload.signature, base64url,
HMAC-SHA256 signed) so it's structurally compatible with any JWT
inspector/debugger even though it's hand-rolled.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from config.settings import settings

_PBKDF2_ITERATIONS = 260_000
_ALGO = "HS256"


# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        salt, digest_hex = hashed.split("$", 1)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)


# --------------------------------------------------------------------------
# JWT (HS256)
# --------------------------------------------------------------------------
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    header = {"alg": _ALGO, "typ": "JWT"}
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + settings.access_token_expire_minutes * 60,
    }
    if extra_claims:
        payload.update(extra_claims)

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(settings.secret_key.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


class InvalidTokenError(Exception):
    pass


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as e:
        raise InvalidTokenError("Malformed token") from e

    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(settings.secret_key.encode(), signing_input, hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception as e:
        raise InvalidTokenError("Malformed signature") from e

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise InvalidTokenError("Signature mismatch")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as e:
        raise InvalidTokenError("Malformed payload") from e

    if payload.get("exp") is not None and time.time() > payload["exp"]:
        raise InvalidTokenError("Token expired")

    return payload

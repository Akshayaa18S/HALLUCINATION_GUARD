"""Shared FastAPI dependencies."""

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import UnauthorizedError
from core.security import InvalidTokenError, decode_access_token
from database.base import get_db
from database.models import User
from services.user_service import UserService


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Reads `Authorization: Bearer <token>`, verifies it, and loads the user.

    Raises UnauthorizedError (-> HTTP 401) if the header is missing, the
    token is invalid/expired, or the user no longer exists.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UnauthorizedError("Missing or malformed Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
    except InvalidTokenError as e:
        raise UnauthorizedError(f"Invalid token: {e}") from e

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedError("Token missing subject")

    user = await UserService(db).get_by_id(user_id)
    if user is None:
        raise UnauthorizedError("User no longer exists")

    return user

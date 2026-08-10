"""Service layer for user signup / authentication."""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import ConflictError, UnauthorizedError
from core.security import hash_password, verify_password
from database.models import User

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: str) -> User | None:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_user(self, email: str, password: str) -> User:
        existing = await self.get_by_email(email)
        if existing is not None:
            raise ConflictError("An account with this email already exists")

        user = User(email=email, hashed_password=hash_password(password))
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        logger.info("Created user %s (%s)", user.id, user.email)
        return user

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise UnauthorizedError("Incorrect email or password")
        return user

"""Authentication endpoints.

POST /api/auth/signup - create an account with a real email + password
POST /api/auth/login  - exchange email + password for a bearer token
GET  /api/auth/me     - resolve the current token to its user
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from core.exceptions import AppError
from core.security import create_access_token
from database.base import get_db
from database.models import User
from models.schemas import LoginRequest, SignupRequest, TokenResponse, UserResponse
from services.user_service import UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(payload: SignupRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    try:
        user = await UserService(db).create_user(payload.email, payload.password)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    token = create_access_token(subject=user.id, extra_claims={"email": user.email})
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    try:
        user = await UserService(db).authenticate(payload.email, payload.password)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e

    token = create_access_token(subject=user.id, extra_claims={"email": user.email})
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)

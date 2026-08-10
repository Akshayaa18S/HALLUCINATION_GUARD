from fastapi import APIRouter

from config.settings import settings
from models.schemas import HealthResponse
from services.llm_service import LLMService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_env=settings.app_env,
    )


@router.get("/health/llm")
async def llm_health_check() -> dict:
    """Separate, slower check - not folded into /health so a slow/unreachable
    Ollama instance never makes basic liveness checks time out."""
    available = await LLMService().is_available()
    return {"ollama_available": available, "model": settings.ollama_model}

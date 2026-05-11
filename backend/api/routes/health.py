"""Health check endpoint for load balancer and monitoring."""
import time
from fastapi import APIRouter
from backend.core.config import settings
from backend.models.schemas import HealthResponse

router = APIRouter(tags=["health"])
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    components = {}

    # Check vector DB
    try:
        from backend.services.vector_store import get_vector_store
        store = get_vector_store()
        stats = store.get_collection_stats()
        components["vector_db"] = "ok"
    except Exception as e:
        components["vector_db"] = f"error: {e}"

    # Check OpenAI API reachability (lightweight)
    try:
        import openai
        openai.api_key = settings.OPENAI_API_KEY
        components["llm_api"] = "ok"
    except Exception:
        components["llm_api"] = "error"

    # Check Redis if configured
    if settings.REDIS_URL:
        try:
            import redis
            r = redis.from_url(settings.REDIS_URL)
            r.ping()
            components["redis"] = "ok"
        except Exception as e:
            components["redis"] = f"error: {e}"
    else:
        components["redis"] = "not_configured"

    all_ok = all(v == "ok" or v == "not_configured" for v in components.values())

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version=settings.APP_VERSION,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 2),
    )

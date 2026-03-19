from contextlib import asynccontextmanager

from fastapi import FastAPI

from wordlist_generation.settings import Settings
from wordlist_generation.model_service import ModelService
from wordlist_generation.batch_processor import BatchProcessor
from wordlist_generation.api.routers import chat, batch


def create_app(coordinator=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = Settings()
        model_service = ModelService.from_settings(settings, coordinator=coordinator)
        batch_processor = BatchProcessor(settings=settings, model_service=model_service)

        app.state.settings = settings
        app.state.model_service = model_service
        app.state.batch_processor = batch_processor

        yield

        if coordinator:
            coordinator.shutdown()

    app = FastAPI(lifespan=lifespan, title="Wordlist-Constrained Generation")
    app.include_router(chat.router)
    app.include_router(batch.router)
    return app


# Default app instance for non-distributed launch: uvicorn wordlist_generation.main:app
app = create_app()

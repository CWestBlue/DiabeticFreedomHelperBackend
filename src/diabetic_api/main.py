"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from diabetic_api.api.routes import chat, dashboard, sessions, upload
from diabetic_api.core.config import get_settings
from diabetic_api.core.exceptions import APIError
from diabetic_api.db.mongo import MongoDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name} v{settings.api_version}")
    print(f"📦 Connecting to MongoDB at {settings.mongo_uri[:20]}...")
    
    MongoDB.connect(settings.mongo_uri, settings.db_name)
    print("✅ MongoDB connected")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    MongoDB.close()
    print("✅ MongoDB connection closed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description="AI-powered diabetic data API with LangGraph agents",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """Handle custom API errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
            },
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        from diabetic_api.agents.llm import get_llm_info
        
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.api_version,
            "mongodb": MongoDB.is_connected(),
            "llm": get_llm_info(settings),
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": settings.app_name,
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/health",
        }
    
    # Include routers
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])
    app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
    app.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
    app.include_router(upload.router, prefix="/upload", tags=["Upload"])
    
    return app


# Create app instance
app = create_app()


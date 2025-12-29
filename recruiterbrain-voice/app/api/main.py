"""
FastAPI Application
Main API server for RecruiterBrain Voice Interview System
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.services.database import (
    initialize_database_connections,
    cleanup_database_connections
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ==========================================
# Lifespan Events
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    Handles startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting RecruiterBrain Voice Interview System...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug: {settings.debug}")
    
    # Initialize database connections
    logger.info("Initializing database connections...")
    success = await initialize_database_connections()
    
    if not success:
        logger.error("❌ Failed to initialize database connections!")
        raise RuntimeError("Database initialization failed")
    
    logger.info("✅ Database connections initialized")
    logger.info(f"✅ Application started on {settings.api_host}:{settings.api_port}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    await cleanup_database_connections()
    logger.info("✅ Cleanup complete")


# ==========================================
# Create FastAPI App
# ==========================================

app = FastAPI(
    title=settings.app_name,
    description="AI-powered phone interview system with voice agents",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)


# ==========================================
# Middleware
# ==========================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Routes
# ==========================================

# Import routers
from app.api.routes import interview as interview_router

# Include routers
app.include_router(
    interview_router.router,
    prefix="/api/v1/interview",
    tags=["interview"]
)


# ==========================================
# Root Endpoints
# ==========================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "environment": settings.app_env
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "postgres": "connected",
        "milvus": "connected",
        "embeddings": "loaded"
    }


# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An error occurred"
        }
    )


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers if not settings.api_reload else 1,
        log_level=settings.log_level.lower()
    )

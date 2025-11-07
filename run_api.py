# run_api.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import api_router
from src.api.middleware import SecurityMiddleware
from config.settings import Settings

def create_app():
    app = FastAPI(
        title="SocialSentinel API",
        description="Social Network Analysis and Content Moderation Platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    security_middleware = SecurityMiddleware()
    app.middleware("http")(security_middleware)
    
    app.include_router(api_router, prefix="/api/v1")
    
    return app

def main():
    settings = Settings()
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        debug=settings.DEBUG
    )

if __name__ == "__main__":
    main()
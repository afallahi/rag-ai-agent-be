import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import for CORS
from api.routes import config as config_routes, health, query


app = FastAPI()

logger = logging.getLogger(__name__)

history = []

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies/authorization headers
    allow_methods=["*"],            # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],            # Allow all headers
)


app.include_router(config_routes.router)
app.include_router(health.router)
app.include_router(query.router)

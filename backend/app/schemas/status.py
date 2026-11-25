"""Response schemas for basic service status endpoints."""

from pydantic import BaseModel


class WelcomeResponse(BaseModel):
    message: str
    project: str


class HealthResponse(BaseModel):
    status: str

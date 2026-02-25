import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.services.inference_service import inference_service

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    python_version: str
    model_loaded: bool


class ModelVersionResponse(BaseModel):
    model_version: str
    model_path: str
    model_loaded: bool
    num_classes: int
    image_size: int
    quantization_hint: str


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        service="Oskín AgTech API",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split()[0],
        model_loaded=inference_service.is_loaded,
    )


@router.get("/model/version", response_model=ModelVersionResponse)
def model_version():
    return ModelVersionResponse(
        model_version=settings.MODEL_VERSION,
        model_path=settings.MODEL_PATH,
        model_loaded=inference_service.is_loaded,
        num_classes=inference_service.num_classes,
        image_size=settings.IMAGE_SIZE,
        quantization_hint="float16",
    )

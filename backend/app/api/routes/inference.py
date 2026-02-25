import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.scan import Scan
from app.models.user import User
from app.services.inference_service import inference_service

logger = logging.getLogger("oskin.inference")
router = APIRouter(prefix="/inference", tags=["inference"])


class PredictionResult(BaseModel):
    rank: int
    class_index: int
    class_name: str
    confidence: float
    confidence_pct: str


class InferenceResponse(BaseModel):
    predictions: List[PredictionResult]
    top_class: str
    top_confidence: float
    model_version: str
    scan_id: Optional[int] = None


@router.post("", response_model=InferenceResponse)
async def run_inference(
    image: UploadFile = File(..., description="Plant leaf image (JPEG/PNG)"),
    field_id: Optional[int] = None,
    save_scan: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not inference_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Inference model not available. "
                "Copy plant_disease.tflite to backend/app/models/ and restart."
            ),
        )

    if image.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported image type: {image.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty image file.",
        )

    try:
        predictions = inference_service.predict(image_bytes, top_k=3)
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    top = predictions[0]
    scan_id = None

    if save_scan:
        scan = Scan(
            user_id=current_user.id,
            field_id=field_id,
            disease_id=None,
            confidence=top["confidence"],
            image_path=image.filename,
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)
        scan_id = scan.id

    from app.core.config import settings

    return InferenceResponse(
        predictions=[PredictionResult(**p) for p in predictions],
        top_class=top["class_name"],
        top_confidence=top["confidence"],
        model_version=settings.MODEL_VERSION,
        scan_id=scan_id,
    )

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ScanCreate(BaseModel):
    field_id: Optional[int] = None
    disease_id: Optional[int] = None
    confidence: Optional[float] = None
    image_path: Optional[str] = None


class ScanOut(BaseModel):
    id: int
    user_id: int
    field_id: Optional[int]
    disease_id: Optional[int]
    confidence: Optional[float]
    image_path: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}

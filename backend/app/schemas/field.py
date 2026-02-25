from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class FieldCreate(BaseModel):
    name: str
    area_hectares: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class FieldOut(BaseModel):
    id: int
    user_id: int
    name: str
    area_hectares: float
    latitude: Optional[float]
    longitude: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}

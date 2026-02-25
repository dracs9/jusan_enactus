from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class WeatherOut(BaseModel):
    id: int
    field_id: int
    temperature: Optional[float]
    humidity: Optional[float]
    precipitation: Optional[float]
    recorded_at: datetime

    model_config = {"from_attributes": True}


class RiskOut(BaseModel):
    field_id: int
    risk_score: int
    risk_level: str
    factors: list[str]

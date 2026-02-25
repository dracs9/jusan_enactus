from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DiseaseOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    symptoms: Optional[str]
    causes: Optional[str]
    treatment_plan: Optional[str]
    prevention: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}

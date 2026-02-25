from typing import Optional
from pydantic import BaseModel


class ProductOut(BaseModel):
    id: int
    supplier_id: int
    name: str
    active_ingredient: Optional[str]
    price: float
    volume: Optional[str]
    description: Optional[str]

    model_config = {"from_attributes": True}

from typing import Optional
from pydantic import BaseModel


class SupplierOut(BaseModel):
    id: int
    name: str
    city: Optional[str]
    contact_phone: Optional[str]
    whatsapp_link: Optional[str]
    external_url: Optional[str]

    model_config = {"from_attributes": True}

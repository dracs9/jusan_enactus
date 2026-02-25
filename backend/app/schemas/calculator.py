from pydantic import BaseModel


class ROIInput(BaseModel):
    area_hectares: float
    expected_yield_t_per_ha: float
    market_price_per_t: float
    loss_percent_without_treatment: float
    treatment_cost: float


class ROIOutput(BaseModel):
    expected_revenue: float
    expected_loss: float
    net_benefit: float
    roi_percentage: float

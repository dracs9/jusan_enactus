from app.schemas.calculator import ROIInput, ROIOutput


def calculate_roi(data: ROIInput) -> ROIOutput:
    total_yield = data.area_hectares * data.expected_yield_t_per_ha
    expected_revenue = total_yield * data.market_price_per_t
    expected_loss = expected_revenue * (data.loss_percent_without_treatment / 100)
    net_benefit = expected_loss - data.treatment_cost

    if data.treatment_cost > 0:
        roi_percentage = (net_benefit / data.treatment_cost) * 100
    else:
        roi_percentage = 0.0

    return ROIOutput(
        expected_revenue=round(expected_revenue, 2),
        expected_loss=round(expected_loss, 2),
        net_benefit=round(net_benefit, 2),
        roi_percentage=round(roi_percentage, 2),
    )

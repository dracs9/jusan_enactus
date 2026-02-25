from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.field import Field
from app.models.weather import WeatherLog
from app.schemas.weather import RiskOut


def get_risk(db: Session, field_id: int, user_id: int) -> RiskOut:
    field = db.query(Field).filter(Field.id == field_id, Field.user_id == user_id).first()
    if not field:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")

    latest_weather = (
        db.query(WeatherLog)
        .filter(WeatherLog.field_id == field_id)
        .order_by(WeatherLog.recorded_at.desc())
        .first()
    )

    risk_score = 0
    factors = []

    if latest_weather:
        temp = latest_weather.temperature or 0
        humidity = latest_weather.humidity or 0
        precipitation = latest_weather.precipitation or 0

        if humidity > 80 and 18 <= temp <= 24:
            risk_score += 70
            factors.append("High humidity with moderate temperature — fungal disease risk elevated")

        if precipitation > 10:
            risk_score += 10
            factors.append("High precipitation detected — waterlogging and root disease risk")

        risk_score = min(risk_score, 100)
    else:
        factors.append("No weather data available for this field")

    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return RiskOut(
        field_id=field_id,
        risk_score=risk_score,
        risk_level=risk_level,
        factors=factors,
    )

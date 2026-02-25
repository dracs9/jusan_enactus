from sqlalchemy.orm import Session
from app.models.scan import Scan
from app.models.disease import Disease
from app.models.field import Field
from app.models.weather import WeatherLog
from app.schemas.chat import ChatResponse
from app.services.risk_service import get_risk
from fastapi import HTTPException


def get_chat_response(db: Session, user_id: int, message: str) -> ChatResponse:
    recent_scan = (
        db.query(Scan)
        .filter(Scan.user_id == user_id, Scan.disease_id.isnot(None))
        .order_by(Scan.created_at.desc())
        .first()
    )

    if recent_scan and recent_scan.disease_id:
        disease = db.query(Disease).filter(Disease.id == recent_scan.disease_id).first()
        if disease:
            confidence_pct = int((recent_scan.confidence or 0) * 100)
            reply = (
                f"Based on your recent scan, {disease.name} was detected with {confidence_pct}% confidence.\n\n"
                f"**Treatment Plan:** {disease.treatment_plan}\n\n"
                f"**Prevention:** {disease.prevention}\n\n"
                f"Apply treatment as soon as possible to minimize crop loss."
            )
            return ChatResponse(reply=reply, context_used="recent_scan")

    user_fields = db.query(Field).filter(Field.user_id == user_id).all()
    high_risk_fields = []
    for field in user_fields:
        try:
            risk = get_risk(db, field.id, user_id)
            if risk.risk_score > 60:
                high_risk_fields.append((field.name, risk.risk_score, risk.risk_level))
        except HTTPException:
            continue

    if high_risk_fields:
        field_warnings = "\n".join(
            [f"- {name}: {score} ({level})" for name, score, level in high_risk_fields]
        )
        reply = (
            f"⚠️ Weather-based risk alert for your fields:\n\n{field_warnings}\n\n"
            f"High humidity and moderate temperatures indicate elevated fungal disease risk. "
            f"Consider applying preventive fungicide treatment and monitoring fields closely. "
            f"Ensure proper drainage to reduce waterlogging risk."
        )
        return ChatResponse(reply=reply, context_used="weather_risk")

    generic_advice = (
        "Here are some general agronomy recommendations for Kazakhstan wheat farming:\n\n"
        "1. **Crop Rotation**: Alternate wheat with legumes or oilseeds every 2-3 years to break disease cycles.\n"
        "2. **Seed Treatment**: Always treat seeds with fungicide before planting to prevent soilborne diseases.\n"
        "3. **Monitoring**: Scout fields weekly during the growing season for early disease detection.\n"
        "4. **Fertilization**: Apply balanced NPK fertilization — excess nitrogen increases disease susceptibility.\n"
        "5. **Harvest Timing**: Harvest at optimal grain moisture to prevent mycotoxin contamination.\n\n"
        "Use the scan feature to detect specific diseases in your fields for targeted advice."
    )
    return ChatResponse(reply=generic_advice, context_used="generic")

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.field import Field
from app.models.weather import WeatherLog
from app.schemas.weather import WeatherOut, RiskOut
from app.services.risk_service import get_risk

router = APIRouter(tags=["weather"])


@router.get("/weather/{field_id}", response_model=List[WeatherOut])
def get_weather(field_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    field = db.query(Field).filter(Field.id == field_id, Field.user_id == current_user.id).first()
    if not field:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")
    logs = db.query(WeatherLog).filter(WeatherLog.field_id == field_id).order_by(WeatherLog.recorded_at.desc()).limit(30).all()
    return logs


@router.get("/risk/{field_id}", response_model=RiskOut)
def get_field_risk(field_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_risk(db, field_id, current_user.id)

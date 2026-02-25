from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.disease import Disease
from app.schemas.disease import DiseaseOut

router = APIRouter(prefix="/diseases", tags=["diseases"])


@router.get("", response_model=List[DiseaseOut])
def list_diseases(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Disease).all()


@router.get("/{disease_id}", response_model=DiseaseOut)
def get_disease(disease_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    disease = db.query(Disease).filter(Disease.id == disease_id).first()
    if not disease:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Disease not found")
    return disease

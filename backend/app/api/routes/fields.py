from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.field import Field
from app.schemas.field import FieldCreate, FieldOut

router = APIRouter(prefix="/fields", tags=["fields"])


@router.post("", response_model=FieldOut, status_code=201)
def create_field(data: FieldCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    field = Field(**data.model_dump(), user_id=current_user.id)
    db.add(field)
    db.commit()
    db.refresh(field)
    return field


@router.get("", response_model=List[FieldOut])
def list_fields(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Field).filter(Field.user_id == current_user.id).all()


@router.get("/{field_id}", response_model=FieldOut)
def get_field(field_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    field = db.query(Field).filter(Field.id == field_id, Field.user_id == current_user.id).first()
    if not field:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")
    return field


@router.delete("/{field_id}", status_code=204)
def delete_field(field_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    field = db.query(Field).filter(Field.id == field_id, Field.user_id == current_user.id).first()
    if not field:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Field not found")
    db.delete(field)
    db.commit()

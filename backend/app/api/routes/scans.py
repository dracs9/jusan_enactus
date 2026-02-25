from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.scan import Scan
from app.schemas.scan import ScanCreate, ScanOut

router = APIRouter(prefix="/scans", tags=["scans"])


@router.post("", response_model=ScanOut, status_code=201)
def create_scan(data: ScanCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    scan = Scan(**data.model_dump(), user_id=current_user.id)
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


@router.get("", response_model=List[ScanOut])
def list_scans(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Scan).filter(Scan.user_id == current_user.id).order_by(Scan.created_at.desc()).all()


@router.get("/{scan_id}", response_model=ScanOut)
def get_scan(scan_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == current_user.id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
    return scan

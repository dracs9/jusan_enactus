from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, Float, String, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    field_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("fields.id", ondelete="SET NULL"), nullable=True)
    disease_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("diseases.id", ondelete="SET NULL"), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="scans")
    field: Mapped[Optional["Field"]] = relationship("Field", back_populates="scans")
    disease: Mapped[Optional["Disease"]] = relationship("Disease", back_populates="scans")

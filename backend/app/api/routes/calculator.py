from fastapi import APIRouter, Depends
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.calculator import ROIInput, ROIOutput
from app.services.calculator_service import calculate_roi

router = APIRouter(prefix="/calculator", tags=["calculator"])


@router.post("/roi", response_model=ROIOutput)
def roi_calculator(data: ROIInput, current_user: User = Depends(get_current_user)):
    return calculate_roi(data)

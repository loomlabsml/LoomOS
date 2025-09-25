# Jobs API

from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def submit_job():
    return {"message": "Job submitted"}
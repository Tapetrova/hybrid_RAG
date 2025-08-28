from fastapi import APIRouter

router = APIRouter()


@router.get("/readiness")
async def readiness():
    return "OK"


@router.get("/liveness")
async def readiness():
    return "OK"

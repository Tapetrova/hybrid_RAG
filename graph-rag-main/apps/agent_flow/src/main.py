from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from apps.agent_flow.src.routers import agent, evaluation
from libs.python.utils import health

# logger.info(f"os.getenv(ENDPOINT_RETRIEVEMENT): {os.getenv('ENDPOINT_RETRIEVEMENT')}")

# from libs.python.databases.database import (
#     db,
#     postgresql_db_url,
# )
# from apps.knowledge_manager.src.utils import logger

app = FastAPI()


# @app.on_event("startup")
# async def startup_event():
#     try:
#         await db.set_bind(postgresql_db_url)
#         await db.gino.create_all()
#         logger.info(f"Connection to DB is Successful")
#     except Exception as e:
#         logger.error(f"Failed to connect to Postgresql: {e}")
#         raise e


# Include routers from the 'routers' module
app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
app.include_router(health.router, prefix="/health", tags=["health"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8096)

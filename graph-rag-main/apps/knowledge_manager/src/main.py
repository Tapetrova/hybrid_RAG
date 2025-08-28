from fastapi import FastAPI
from apps.knowledge_manager.src.routers import (
    retrievement,
    building,
    from_s3_to_sql_table_data_uploading_route,
)
from libs.python.databases.database import (
    db,
    postgresql_db_url,
)
from apps.knowledge_manager.src.utils import logger
from libs.python.utils import health

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    try:
        await db.set_bind(postgresql_db_url)
        await db.gino.create_all()

        logger.info(f"[GINO] Connection to DB is Successful")
    except Exception as e:
        logger.error(f"Failed to connect to Postgresql: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    try:
        await db.pop_bind().close()
        logger.info(f"[GINO] Connection to DB is closed Successfully!")
    except Exception as e:
        logger.error(f"Failed to close the connection to Postgresql: {e}")
        raise e


# Include routers from the 'routers' module
app.include_router(retrievement.router, prefix="/retrievement", tags=["retrievement"])
app.include_router(building.router, prefix="/building", tags=["building"])
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(
    from_s3_to_sql_table_data_uploading_route.router,
    prefix="/from_s3_to_sql_table_data_uploading",
    tags=["from_s3_to_sql_table_data_uploading"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8098)

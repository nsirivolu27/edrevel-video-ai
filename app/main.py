from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import database
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="Edrevel Object Interaction Analyzer",
    description="Small FastAPI project for object interaction analysis in videos.",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(router)

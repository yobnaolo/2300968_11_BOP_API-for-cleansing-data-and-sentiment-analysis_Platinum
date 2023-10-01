from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import sqlite3 as sql

app = FastAPI()

@app.get("/")
async def index():

    return FileResponse('templates/home.html')

from routers import cleansing
from routers import sentiment
from routers import database

app.include_router(cleansing.router, tags=["Cleansing API"])
app.include_router(sentiment.router, tags=["Sentiment API"])
# app.include_router(database.router, tags=["Database API"])



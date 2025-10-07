from fastapi import FastAPI, APIRouter, HTTPException, Path, Query, Body, status, Depends, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import uuid4
import shutil, io, os, asyncio

from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext


SECRET_KEY = "demo-secret-key-change-me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI(title="FastAPI Daily Series (35 endpoints)")


app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)
week1 = APIRouter(prefix="/week1", tags=["week1 - basics"])


class Item(BaseModel):
id: Optional[int] = None
name: str
price: float
description: Optional[str] = None


@week1.get("/day01_hello")
async def hello_world():
return {"message": "Hello, FastAPI demo!"}


@week1.get("/day02_path/{item_id}")
async def get_item_path(item_id: int = Path(...)):
return {"item_id": item_id}


@week1.get("/day03_query")
async def search(query: Optional[str] = Query(None), limit: int = 10):
return {"query": query, "limit": limit}


@week1.post("/day04_body", status_code=201)
async def create_item(item: Item):
item.id = 1
return item


@week1.post("/day05_response_model", response_model=Item)
async def create_item_response(item: Item):
item.id = 2
return item


@week1.get("/day06_status")
async def custom_status():
return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"detail": "Accepted for processing"})


@week1.get("/day07_structure")
async def project_structure():
return {"note": "Use routers, models, and db modules for structure."}


app.include_router(week1)

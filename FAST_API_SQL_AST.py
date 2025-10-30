"""
FastAPI service for SQL-to-AST model inference
File: fastapi_sql_to_ast_app.py

This single-file template includes:
- FastAPI app with endpoints: /health, /predict (single), /batch (bulk), /models
- Robust request validation using Pydantic
- Structured error handling and logging
- A ModelLoader class that attempts to load a HuggingFace transformers + PEFT/LoRA model, and falls
  back to a stub for local development / tests.
- Dockerfile & requirements.txt embedded as comments (ready to copy out)
- Example pytest-based tests at the end of the file (commented)

Notes:
- Replace the model_path and tokenizer_path with your fine-tuned LoRA CodeT5 model artifacts.
- This file is intentionally dependency-light at runtime: the model loader will not crash the app if
  transformer libs are missing — it will expose a predictable stub so the API can be tested.

License: MIT 
"""

# -----------------------------
# Requirements (copy to requirements.txt)
# -----------------------------
# fastapi
# uvicorn[standard]
# pydantic
# transformers>=4.30.0   # optional: only if you will load the real model
# accelerate               # optional
# peft                    # optional: for LoRA weights
# sentencepiece           # if using some tokenizers
# torch>=1.12             # or the appropriate backend; optional
# python-multipart        # for form uploads if needed
# httpx[http2]            # for testing

# -----------------------------
# Dockerfile (copy to Dockerfile)
# -----------------------------
# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . /app
# ENV PYTHONUNBUFFERED=1
# CMD ["uvicorn", "fastapi_sql_to_ast_app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]

# -----------------------------
# The code
# -----------------------------
import time
import logging
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from functools import lru_cache

# Optional imports for real model loading. Wrapped in try/except so the app can run without them.
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# -----------------------------
# Logger configuration
# -----------------------------
logger = logging.getLogger("sql_to_ast_api")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(request_id)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Helper to attach request_id to log records
class RequestIDAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        request_id = self.extra.get("request_id")
        return msg, {**kwargs, "extra": {"request_id": request_id}}


# -----------------------------
# Pydantic models
# -----------------------------
class PredictRequest(BaseModel):
    sql: str = Field(..., description="SQL query to translate to AST", example="SELECT id, name FROM users WHERE age > 18")
    dialect: Optional[str] = Field(None, description="SQL dialect hint (e.g., mysql, postgres)")
    max_tokens: Optional[int] = Field(512, ge=16, le=4096, description="Max tokens/length for model generation")

    @validator("sql")
    def sql_must_not_be_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("sql must be a non-empty string")
        if len(v) > 10000:
            raise ValueError("sql too long")
        return v.strip()


class ASTResponse(BaseModel):
    ast: Dict[str, Any] = Field(..., description="AST represented as a JSON-serializable object")
    tokens: Optional[List[str]] = Field(None, description="Optional tokenized output from model")
    confidence: Optional[float] = Field(None, description="Optional model confidence score (0-1)")


class BulkPredictRequest(BaseModel):
    items: List[PredictRequest]

    @validator("items")
    def non_empty_items(cls, v):
        if not v:
            raise ValueError("items must contain at least one request")
        if len(v) > 100:
            raise ValueError("batch size too large; max 100")
        return v


# -----------------------------
# Model loader abstraction
# -----------------------------
class ModelLoader:
    """Loads a seq2seq model and tokenizer (if available). Provides a generate(ast) method.

    If transformers/peft are not installed or model_path is None, the loader returns a stub
    implementation that deterministically converts SQL into a fake AST for testing the API.
    """

    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.ready = False

        if HAS_TRANSFORMERS and model_path:
            try:
                logger.info(f"Attempting to load model from {model_path}", extra={"request_id": "loader"})
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                if device.startswith("cuda") and torch.cuda.is_available():
                    self.model.to(device)
                self.ready = True
                logger.info("Model loaded successfully", extra={"request_id": "loader"})
            except Exception as e:
                logger.exception("Failed to load real model; falling back to stub", extra={"request_id": "loader"})
                self._setup_stub()
        else:
            self._setup_stub()

    def _setup_stub(self):
        # stub: deterministic SQL parser placeholder — returns a JSON-like structure
        self.model = None
        self.tokenizer = None
        self.ready = True
        logger.info("Using stub model for SQL->AST (no transformers installed or path not provided)", extra={"request_id": "loader"})

    def _stub_parse(self, sql: str, dialect: Optional[str] = None) -> Dict[str, Any]:
        # naive, deterministic fake AST — useful for testing API wiring
        tokens = [t for t in sql.replace('\n', ' ').split(' ') if t]
        ast = {
            "type": "Select" if sql.strip().lower().startswith("select") else "Unknown",
            "raw": sql,
            "clauses": [t.upper() for t in tokens[:10]],
            "dialect_hint": dialect,
        }
        return {"ast": ast, "tokens": tokens[:512], "confidence": 0.42}

    def generate(self, sql: str, dialect: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Run model inference (synchronously). If a real model is loaded, it will be used; otherwise stub."""
        if self.model is None:
            return self._stub_parse(sql, dialect)

        # Example real model flow (pseudo-code). Tune generation parameters as needed.
        try:
            inputs = self.tokenizer(sql, return_tensors="pt", truncation=True, padding=True)
            if self.device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            gen = self.model.generate(**inputs, max_length=max_tokens or 512)
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            # Simple post-processing: in practice you'd convert the output string to JSON AST safely
            return {"ast": {"generated": decoded}, "tokens": decoded.split(), "confidence": 0.9}
        except Exception as e:
            logger.exception("Model inference failure", extra={"request_id": "model"})
            raise


# singleton model loader (fast startup cache)
@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    # In production, set MODEL_PATH/TOKENIZER via environment or config management
    return ModelLoader(model_path=None, tokenizer_path=None, device="cpu")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="SQL-to-AST Inference API",
    version="0.1.0",
    description="Translate SQL text into AST JSON using a fine-tuned CodeT5+LoRA model."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def add_logging_and_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    adapter = RequestIDAdapter(logger, {"request_id": request_id})
    start = time.time()
    adapter.info(f"start {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as exc:
        adapter.exception("unhandled exception during request")
        raise
    duration = time.time() - start
    adapter.info(f"end {request.method} {request.url.path} status={response.status_code} duration={duration:.3f}s")
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}", extra={"request_id": "err"})
    return fastapi.responses.JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.on_event("startup")
async def startup_event():
    # eager model loading — if you prefer lazy loading, remove this
    get_model_loader()
    logger.info("application startup complete", extra={"request_id": "startup"})


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health", summary="Health check")
async def health_check():
    """Simple health endpoint with model readiness state."""
    loader = get_model_loader()
    return {"status": "ok", "model_ready": loader.ready}


@app.get("/models", summary="List available models / metadata")
async def list_models():
    # In a real deployment you might query a model registry
    loader = get_model_loader()
    meta = {
        "loaded": bool(loader.model),
        "using_stub": loader.model is None,
        "model_path": loader.model_path,
    }
    return {"models": [meta]}


@app.post("/predict", response_model=ASTResponse, status_code=status.HTTP_200_OK, summary="Translate a single SQL query to AST")
async def predict(req: PredictRequest = Body(..., example={"sql": "SELECT * FROM users WHERE id = 1", "dialect": "postgres"})):
    loader = get_model_loader()
    try:
        raw = loader.generate(req.sql, dialect=req.dialect, max_tokens=req.max_tokens)
        ast = raw.get("ast")
        tokens = raw.get("tokens")
        confidence = raw.get("confidence")
        return ASTResponse(ast=ast, tokens=tokens, confidence=confidence)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="model inference failed")


@app.post("/batch", summary="Batch translate multiple SQL queries", status_code=status.HTTP_200_OK)
async def batch_predict(req: BulkPredictRequest):
    loader = get_model_loader()
    results = []
    # Synchronous loop — for performance in production consider batching or async concurrency
    for item in req.items:
        try:
            raw = loader.generate(item.sql, dialect=item.dialect, max_tokens=item.max_tokens)
            results.append({"input_sql": item.sql, "ast": raw.get("ast"), "confidence": raw.get("confidence")})
        except Exception as e:
            results.append({"input_sql": item.sql, "error": "inference_failed"})
    return {"results": results}


# -----------------------------
# If invoked directly, start uvicorn (useful for local dev)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_sql_to_ast_app:app", host="0.0.0.0", port=8080, log_level="info")


# -----------------------------
# Example tests (pytest + httpx)
# Save this as test_api.py and run `pytest -q`
# -----------------------------
# import pytest
# from fastapi_sql_to_ast_app import app
# from httpx import AsyncClient
#
# @pytest.mark.asyncio
# async def test_health():
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         r = await ac.get("/health")
#         assert r.status_code == 200
#         assert r.json()["status"] == "ok"
#
# @pytest.mark.asyncio
# async def test_predict_stub():
#     payload = {"sql": "SELECT id FROM products WHERE price > 100"}
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         r = await ac.post("/predict", json=payload)
#         assert r.status_code == 200
#         data = r.json()
#         assert "ast" in data
#
# @pytest.mark.asyncio
# async def test_batch_limit():
#     items = [{"sql": "SELECT 1"}] * 2
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         r = await ac.post("/batch", json={"items": items})
#         assert r.status_code == 200


import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model import load_model, get_model
from preprocess import preprocess_for_inference
from schemas import PredictionResponse
from download_weights import download_weights

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_weights()   # ← add this line before load_model()
    load_model()
    yield

# ── Logging (no PHI ever logged) ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Rate Limiter ────────────────────────────────────────────────────────────
# Prevent abuse. Medical APIs should not be hammered with automated requests.
limiter = Limiter(key_func=get_remote_address)

# ── Lifespan: load model once on startup ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()    # Load before serving first request
    yield
    # Cleanup on shutdown (if any resources to release)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MammoAI — Breast Cancer Screening API",
    description="Binary classification of mammogram images: Malignant vs Benign/Normal. "
                "FOR RESEARCH AND DECISION-SUPPORT ONLY. Not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ────────────────────────────────────────────────────────────────────
# In production, replace allow_origins with your exact frontend URL
# Open CORS in development only
import os
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "https://your-firebase-app.web.app"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Request ID middleware ───────────────────────────────────────────────────
# Assigns a UUID to every request for traceability without storing patient data
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}

# ── Prediction endpoint ────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("20/minute")  # Max 20 predictions per minute per IP
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Accepts a mammogram image (DICOM, PNG, JPEG).
    Returns malignancy classification and confidence score.
    
    Privacy: The uploaded image is processed in memory and NEVER written to disk.
    No patient metadata is logged or stored.
    """
    request_id = request.state.request_id
    
    # ── Validate file ──────────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {ALLOWED_EXTENSIONS}"
        )
    
    file_bytes = await file.read()
    
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
    
    # Log only: request ID, file extension, file size — NEVER filename or content
    logger.info(f"[{request_id}] Predict | ext={suffix} | size={len(file_bytes)} bytes")
    
    # ── Preprocess & Infer ─────────────────────────────────────────────────
    try:
        tensor = preprocess_for_inference(file_bytes, file.filename or "image.png")
        tensor = tensor.to(DEVICE)
    except Exception as e:
        logger.warning(f"[{request_id}] Preprocessing failed: {type(e).__name__}")
        raise HTTPException(status_code=422, detail="Could not process image. "
                            "Ensure it is a valid DICOM, PNG, or JPEG mammogram.")
    
    model, config = get_model()
    
    with torch.no_grad():
        logit = model(tensor)
        prob_malignant = float(torch.sigmoid(logit).cpu().item())
    
    prob_benign = 1.0 - prob_malignant
    classification = "Malignant" if prob_malignant >= config["threshold"] else "Benign/Normal"
    
    logger.info(f"[{request_id}] Result: {classification} | prob={prob_malignant:.4f}")
    
    return PredictionResponse(
        classification=classification,
        confidence=max(prob_malignant, prob_benign),
        malignant_probability=round(prob_malignant, 4),
        benign_probability=round(prob_benign, 4),
        warning=(
            "This result is generated by an AI model for research and decision-support "
            "purposes only. It must not replace clinical diagnosis by a qualified "
            "radiologist. Always confirm with expert review."
        ),
    )
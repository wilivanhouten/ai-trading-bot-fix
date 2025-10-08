#!/usr/bin/env python3
"""
AI Trading Bot - Backend with Groq API
Deploy: Railway.app / Render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import re
from typing import Optional
from groq import Groq

app = FastAPI(
    title="AI Trading Bot API - Groq",
    description="Professional MQL4/MQ5/Pine Script Generator",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key via Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not set!")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama-3.3-70b-versatile"

class GenerateRequest(BaseModel):
    prompt: str
    platform: str = "auto"
    type: str = "auto"
    complexity: str = "auto"

class HealthResponse(BaseModel):
    status: str
    model: str
    message: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "AI Trading Bot API - Powered by Groq",
        "model": MODEL_NAME,
        "status": "online" if groq_client else "offline"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    if not groq_client:
        return HealthResponse(
            status="error",
            model="offline",
            message="❌ GROQ_API_KEY not configured"
        )
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        message="✅ Groq API ready!"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

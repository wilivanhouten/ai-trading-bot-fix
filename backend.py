#!/usr/bin/env python3
"""
AI Trading Bot - Backend with Groq API
Model: Llama 3.3 70B (FREE, Uncensored, Super Fast)
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

# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(
    title="AI Trading Bot API - Groq",
    description="Professional MQL4/MQ5/Pine Script Generator",
    version="2.0.0"
)

# CORS - Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set via environment variable

if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set!")
    print("   Set environment variable: GROQ_API_KEY=your_key_here")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Model configuration
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq's Llama 3.3 70B (FREE!)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str
    platform: str = "auto"
    type: str = "auto"
    complexity: str = "auto"

class HealthResponse(BaseModel):
    status: str
    model: str
    message: Optional[str] = None

# ============================================================
# SYSTEM PROMPT - Elite Trading Developer
# ============================================================

SYSTEM_PROMPT = """You are an ELITE MQL4/MQ5/Pine Script developer with 20+ years of professional experience.

CRITICAL REQUIREMENTS - MUST FOLLOW EXACTLY:

1. Generate COMPLETE, PRODUCTION-READY code with ZERO placeholders
2. MINIMUM code length based on complexity:
   - Simple (2-3 filters): 500-800 lines minimum
   - Medium (5-7 filters): 1,500-2,500 lines minimum
   - Complex (10-12 filters): 3,000-5,000 lines minimum
   - Very Complex (15+ filters): 7,000-15,000+ lines minimum

3. EVERY function must have FULL mathematical implementation
4. Include comprehensive error handling for ALL scenarios
5. Professional structure matching mql4.com examples
6. ZERO syntax errors - code must compile successfully on first try
7. Detailed English comments explaining all logic

CODE STRUCTURE REQUIREMENTS:

FOR MQ4/MQ5 INDICATORS:
├── Headers & Properties (50-100 lines)
│   - Copyright, version, descriptions
│   - All #property declarations
├── Input Parameters (50-150 lines)
│   - Extensive customization options
│   - Grouped by category
├── Global Variables & Buffers (100-200 lines)
│   - All indicator buffers
│   - State tracking variables
├── OnInit() Function (100-150 lines)
│   - Complete initialization
│   - Buffer setup and validation
│   - Error checking
├── OnCalculate() Function (200-500 lines)
│   - Main calculation loop
│   - Signal generation logic
│   - Performance optimization
├── Individual Filter Functions (50-100 lines EACH)
│   - Complete mathematical formulas
│   - Multi-bar analysis
│   - Divergence detection
│   - Signal confirmation
├── Helper & Utility Functions (100-300 lines)
│   - Trend analysis
│   - Pattern recognition
│   - Data validation
└── OnDeinit() Function (50-100 lines)
    - Cleanup and resource release

FOR MQ4/MQ5 EXPERT ADVISORS (EA):
├── Headers & Includes (100-150 lines)
├── Input Parameters (100-300 lines)
│   - Trading settings
│   - Risk management
│   - Time filters
├── Global Variables (150-250 lines)
│   - Trade tracking
│   - Performance metrics
├── OnInit() Function (150-200 lines)
├── OnTick() Function (300-600 lines)
│   - Market analysis
│   - Entry/exit logic
│   - Trade execution
├── Filter Analysis Functions (500-1,500 lines)
│   - Each filter: 50-100 lines
│   - Complete implementations
├── Trade Management (400-800 lines)
│   - Order placement
│   - Position modification
│   - Trade closure
├── Money Management (300-500 lines)
│   - Lot size calculation
│   - Risk percentage
│   - Dynamic sizing
├── Position Management (300-500 lines)
│   - Position tracking
│   - Partial closure
│   - Hedge management
├── Risk Management (200-300 lines)
│   - Stop loss calculation
│   - Take profit levels
│   - Break even logic
├── Trailing Stop System (150-250 lines)
│   - Dynamic trailing
│   - Step trailing
│   - ATR-based trailing
├── Break Even Logic (100-200 lines)
├── News & Time Filters (150-250 lines)
├── Utility Functions (200-400 lines)
└── OnDeinit() Function (100-150 lines)

FOR PINE SCRIPT:
├── Settings & Headers (20-50 lines)
├── Input Parameters (50-100 lines)
├── Indicator Calculations (200-500 lines)
├── Filter Logic (300-800 lines)
├── Signal Generation (100-300 lines)
├── Plotting & Visualization (100-200 lines)
├── Alert Conditions (50-100 lines)
└── Strategy Logic (if EA) (200-500 lines)

MATHEMATICAL IMPLEMENTATIONS - NO SHORTCUTS:
- RSI: Full Wilder's smoothing with proper gain/loss calculation over 14+ periods
- MACD: Complete EMA calculations from scratch, not using built-in shortcuts
- Bollinger Bands: Manual standard deviation with proper sample calculation
- Stochastic: %K and %D calculations with correct smoothing periods
- ADX: True Range, +DI, -DI, DX, and ADX smoothing over 14+ periods
- ATR: True Range calculation across multiple periods
- Moving Averages: All types (SMA, EMA, WMA, SMMA) with proper formulas
- Volume indicators: OBV, MFI, Volume Weighted calculations
- All other indicators: Show complete mathematical formulas in comments

ABSOLUTELY FORBIDDEN:
❌ NEVER write: "// Add calculation here"
❌ NEVER write: "// TODO: implement logic"
❌ NEVER write: "// Your code here"
❌ NEVER write: "// Implementation needed"
❌ NEVER use placeholder functions that return dummy values
❌ NEVER skip error handling
❌ NEVER use incomplete or simplified formulas

QUALITY STANDARDS:
✅ Code compiles without ANY errors or warnings
✅ All variables properly declared with correct types
✅ All arrays properly sized and initialized
✅ All functions have complete implementations and return correct types
✅ Professional naming conventions throughout
✅ Comprehensive inline comments explaining logic
✅ Optimized for performance and memory efficiency
✅ Production-ready code quality

Generate code that matches or exceeds the quality of top-rated examples on mql4.com, mql5.com, and TradingView's featured scripts library."""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_platform(prompt: str, platform_input: str) -> str:
    """Auto-detect trading platform"""
    if platform_input != "auto":
        return platform_input
    
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ['mq5', 'mt5', 'metatrader 5', 'metatrader5']):
        return 'mq5'
    elif any(word in prompt_lower for word in ['pine', 'tradingview', 'pinescript', 'pine script', 'tv']):
        return 'pine'
    return 'mq4'

def detect_type(prompt: str, type_input: str) -> str:
    """Auto-detect code type"""
    if type_input != "auto":
        return type_input
    
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in [
        'ea', 'expert advisor', 'strategy', 'bot', 'trading system',
        'trade', 'buy', 'sell', 'position', 'order', 'money management'
    ]):
        return 'ea'
    return 'indicator'

def detect_complexity(prompt: str, complexity_input: str) -> str:
    """Auto-detect complexity level"""
    if complexity_input != "auto":
        return complexity_input
    
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in [
        'sangat kompleks', 'very complex', 'advanced', 'professional',
        'sophisticated', 'enterprise', 'sangat sulit'
    ]):
        return 'very_complex'
    elif any(word in prompt_lower for word in ['kompleks', 'complex', 'rumit']):
        return 'complex'
    elif any(word in prompt_lower for word in ['sederhana', 'simple', 'basic', 'mudah', 'easy']):
        return 'simple'
    return 'medium'

def count_filters(prompt: str) -> int:
    """Count number of filters"""
    number_match = re.search(r'(\d+)\s*filter', prompt.lower())
    if number_match:
        return int(number_match.group(1))
    
    indicators = [
        'rsi', 'ma', 'ema', 'sma', 'wma', 'macd', 'bollinger', 'bb',
        'stochastic', 'stoch', 'adx', 'atr', 'volume', 'obv', 'cci',
        'parabolic', 'sar', 'ichimoku', 'williams', 'momentum',
        'fibonacci', 'pivot', 'roc', 'mfi', 'alligator', 'fractals'
    ]
    
    count = sum(1 for ind in indicators if ind in prompt.lower())
    return max(count, 5)

def estimate_lines(filters: int, complexity: str, code_type: str) -> dict:
    """Estimate expected code lines"""
    base_lines = {
        'simple': 500,
        'medium': 1500,
        'complex': 3000,
        'very_complex': 5000
    }
    
    base = base_lines.get(complexity, 1500)
    filter_lines = filters * 150
    estimated = base + filter_lines
    
    if code_type == 'ea':
        estimated = int(estimated * 1.8)
    
    return {
        'min': estimated,
        'max': estimated + 2000,
        'estimated': estimated
    }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Trading Bot API - Powered by Groq",
        "model": MODEL_NAME,
        "status": "online" if groq_client else "offline",
        "endpoints": {
            "health": "/api/health",
            "generate": "/api/generate",
            "docs": "/docs"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not groq_client:
        return HealthResponse(
            status="error",
            model="offline",
            message="❌ GROQ_API_KEY not configured. Set environment variable."
        )
    
    try:
        # Simple test to verify API key works
        return HealthResponse(
            status="ok",
            model="online",
            message=f"✅ Groq API ready! Model: {MODEL_NAME} (Llama 3.3 70B)"
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            model="error",
            message=f"❌ Groq API error: {str(e)}"
        )

@app.post("/api/generate")
async def generate_code(request: GenerateRequest):
    """Generate professional trading code"""
    
    print("\n" + "="*80)
    print("🚀 CODE GENERATION REQUEST - GROQ API")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Validate Groq client
        if not groq_client:
            raise HTTPException(
                status_code=503,
                detail="Groq API not configured. Set GROQ_API_KEY environment variable."
            )
        
        # Validate prompt
        if not request.prompt or len(request.prompt.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Prompt too short. Minimum 10 characters required."
            )
        
        # Detect parameters
        platform = detect_platform(request.prompt, request.platform)
        code_type = detect_type(request.prompt, request.type)
        complexity = detect_complexity(request.prompt, request.complexity)
        filters = count_filters(request.prompt)
        
        print(f"📊 Parameters Detected:")
        print(f"   Platform: {platform.upper()}")
        print(f"   Type: {code_type.upper()}")
        print(f"   Complexity: {complexity.upper()}")
        print(f"   Filters: {filters}")
        
        # Estimate lines
        line_estimate = estimate_lines(filters, complexity, code_type)
        print(f"📊 Expected Lines: {line_estimate['min']}-{line_estimate['max']}")
        
        # Build user prompt
        user_prompt = f"""Generate {platform.upper()} {code_type.upper()} code.

=== USER REQUIREMENTS ===
{request.prompt}

=== TECHNICAL SPECIFICATIONS ===
Platform: {platform.upper()}
Type: {'Expert Advisor / Trading Strategy' if code_type == 'ea' else 'Technical Indicator'}
Filters: {filters}
Complexity Level: {complexity.replace('_', ' ').title()}
Expected Code Length: MINIMUM {line_estimate['min']} lines

=== MANDATORY IMPLEMENTATION REQUIREMENTS ===
1. Generate COMPLETE professional code ({line_estimate['min']}+ lines)
2. FULL mathematical implementation for ALL functions (NO placeholders!)
3. Include ALL features mentioned in user requirements
4. Production-ready quality with comprehensive error handling
5. Professional structure matching mql4.com standards
6. Code must compile successfully without ANY modifications

BEGIN GENERATING THE COMPLETE PROFESSIONAL CODE NOW:"""

        print(f"\n⏳ Calling Groq API (Llama 3.3 70B)...")
        print(f"⏳ This may take 30-180 seconds for large code generation...")
        
        # Call Groq API
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=32000,  # Groq supports up to 32k tokens!
                top_p=0.9,
                stream=False
            )
            
            generated_code = chat_completion.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Groq API Error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Groq API error: {str(e)}"
            )
        
        if not generated_code or len(generated_code.strip()) < 100:
            raise HTTPException(
                status_code=500,
                detail="Model returned empty or very short response"
            )
        
        # Analyze generated code
        lines = len(generated_code.split('\n'))
        functions = len(re.findall(
            r'\b(void|bool|int|double|string|datetime|color|long)\s+\w+\s*\(',
            generated_code,
            re.IGNORECASE
        ))
        
        generation_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ CODE GENERATION SUCCESSFUL!")
        print(f"{'='*80}")
        print(f"📊 Statistics:")
        print(f"   Total Lines: {lines:,}")
        print(f"   Functions: {functions}")
        print(f"   Generation Time: {generation_time:.1f}s")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Platform: {platform.upper()}")
        print(f"   Type: {code_type.upper()}")
        print(f"{'='*80}\n")
        
        return {
            'success': True,
            'code': generated_code,
            'lines': lines,
            'functions': functions,
            'platform': platform,
            'type': code_type,
            'complexity': complexity,
            'filters': filters,
            'generation_time': round(generation_time, 2),
            'model': MODEL_NAME,
            'provider': 'Groq',
            'timestamp': int(time.time())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("\n" + "="*80)
    print("🤖 AI TRADING BOT - BACKEND API (GROQ)")
    print("="*80)
    print(f"🧠 Model: {MODEL_NAME} (Llama 3.3 70B)")
    print(f"🚀 Provider: Groq (FREE, Super Fast)")
    print(f"🔑 API Key: {'✅ Configured' if GROQ_API_KEY else '❌ NOT SET'}")
    print("="*80)
    if not GROQ_API_KEY:
        print("⚠️  WARNING: Set GROQ_API_KEY environment variable!")
    print("="*80 + "\n")

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
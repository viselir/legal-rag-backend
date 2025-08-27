# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
from fastapi.responses import PlainTextResponse, JSONResponse

import os
import sys
from pathlib import Path

# הוספת נתיב כדי לייבא את legal-rag/rag_service.py
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
LEGAL_RAG_DIR_DEFAULT = PROJECT_ROOT / "legal-rag"
sys.path.append(str(LEGAL_RAG_DIR_DEFAULT))

from rag_service import answer_question, ensure_index_built

app = FastAPI()

class HistoryMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[HistoryMsg]] = None
    stream: bool = False
    top_k: int = 4

@app.get("/health")
def health():
    return {"ok": True}

# אינדוקס אוטומטי חד-פעמי על אתחול השרת
@app.on_event("startup")
def _warmup():
    force = os.getenv("RAG_FORCE_REINDEX", "0") in ("1", "true", "True", "yes")
    try:
        ensure_index_built(force=force)
        print("[startup] Index is ready.")
    except Exception as e:
        print("[startup] Index build failed:", repr(e))

# מענה טקסט (מה שה-UI צורך)
@app.post("/chat", response_class=PlainTextResponse)
def chat(req: ChatRequest):
    text, _ = answer_question(req.prompt, [m.model_dump() for m in (req.history or [])], req.top_k)
    return text

# (אופציונלי) מענה JSON עם מקורות, אם בעתיד תרצה
@app.post("/chat_json")
def chat_json(req: ChatRequest):
    text, sources = answer_question(req.prompt, [m.model_dump() for m in (req.history or [])], req.top_k)
    return JSONResponse({"text": text, "sources": sources})

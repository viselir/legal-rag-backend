# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
from fastapi.responses import PlainTextResponse, JSONResponse

# --- הופכים את legal-rag לייביאבל ע"י הוספת הנתיב ---
import sys
from pathlib import Path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
LEGAL_RAG_DIR = PROJECT_ROOT / "legal-rag"
sys.path.append(str(LEGAL_RAG_DIR))

# --- מייבאים את השירות שלך ---
from rag_service import answer_question  # ← זה הקובץ שיצרנו עכשיו

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

# מסלול טקסט (ה-UI שלך כבר צורך אותו)
@app.post("/chat", response_class=PlainTextResponse)
def chat(req: ChatRequest):
    text, _ = answer_question(req.prompt, [m.model_dump() for m in (req.history or [])], req.top_k)
    return text

# (אופציונלי) מסלול שמחזיר גם מקורות כ-JSON
@app.post("/chat_json")
def chat_json(req: ChatRequest):
    text, sources = answer_question(req.prompt, [m.model_dump() for m in (req.history or [])], req.top_k)
    return JSONResponse({"text": text, "sources": sources})

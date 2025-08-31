# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Literal, Optional, Callable, Tuple, TYPE_CHECKING
from fastapi.middleware.cors import CORSMiddleware
import os, sys, traceback, importlib, importlib.util
from pathlib import Path
from dotenv import load_dotenv

# --- Paths & .env ---
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
LEGAL_RAG_DIR = Path(os.getenv("LEGAL_RAG_DIR", PROJECT_ROOT / "legal-rag")).resolve()

if str(LEGAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(LEGAL_RAG_DIR))

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(LEGAL_RAG_DIR / ".env")

# --- Import rag_service robustly (ללא from-import כדי שה-IDE לא יסמן) ---
try:
    rs = importlib.import_module("rag_service")
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("rag_service", str(LEGAL_RAG_DIR / "rag_service.py"))
    if not spec or not spec.loader:
        raise
    rs = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(rs)                 # type: ignore[assignment]

answer_question = getattr(rs, "answer_question", None)
ensure_index_built = getattr(rs, "ensure_index_built", None)

if not callable(answer_question) or not callable(ensure_index_built):
    available = ", ".join(n for n, o in rs.__dict__.items() if callable(o))
    raise RuntimeError(
        f"rag_service נטען מ־{getattr(rs,'__file__','?')} אך לא נמצאו answer_question/ensure_index_built. "
        f"פונקציות זמינות: {available}"
    )

# טיפוסי עזר ל-IDE (לא משפיע על ריצה)
if TYPE_CHECKING:
    answer_question: Callable[[str, List[dict] | None, int], Tuple[str, List[dict]]]
    ensure_index_built: Callable[[bool], None]

# --- FastAPI app ---
app = FastAPI(title="Legal-RAG Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class HistoryMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[HistoryMsg]] = None
    top_k: int = 4

@app.get("/health")
def health():
    return {"ok": True}

# אפשר להשאיר כך; האזהרה על on_event היא דפרקציה בלבד
@app.on_event("startup")
def _warmup():
    try:
        force = os.getenv("RAG_FORCE_REINDEX", "0").lower() in ("1", "true", "yes")
        ensure_index_built(force=force)
        print("[startup] Index is ready.")
    except Exception as e:
        print("[startup] Index build failed:", repr(e))

@app.post("/chat", response_class=PlainTextResponse)
def chat(req: ChatRequest):
    try:
        text, _ = answer_question(req.prompt, [m.model_dump() for m in (req.history or [])], req.top_k)
        return text
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_json")
def chat_json(req: ChatRequest):
    try:
        text, sources = answer_question(req.prompt, [m.model_dump() for m in (req.history or [])], req.top_k)
        return JSONResponse({"text": text, "sources": sources})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

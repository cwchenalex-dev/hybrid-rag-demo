from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .indexer import load_hybrid_index
from .routes import router

app = FastAPI(title="Hybrid RAG Knowledge Base Q&A Bot")
app.include_router(router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return index_html.read_text(encoding="utf-8")
    return "<h1>UI index.html not found</h1>"


@app.on_event("startup")
def load_persisted_index():
    try:
        load_hybrid_index()
    except Exception as exc:
        print(f"[hybrid_kb] Skipping persisted load: {exc}", flush=True)
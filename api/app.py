import logging
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from api.routes import health, query, database, graph, config

logger = logging.getLogger(__name__)

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"

app = FastAPI(
    title="SQL Agent API",
    description="LangGraph-powered SQL Agent for natural-language database queries.",
    version="1.0.0",
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query.router)
app.include_router(database.router)
app.include_router(graph.router)
app.include_router(config.router)


# ── Frontend ───────────────────────────────────────────────────────────────────
# Serve Vite production build from frontend/dist/ if available,
# otherwise fall back to frontend/index.html (legacy)
DIST_DIR = FRONTEND_DIR / "dist"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the frontend single-page application."""
    if (DIST_DIR / "index.html").exists():
        return FileResponse(str(DIST_DIR / "index.html"))
    index = FRONTEND_DIR / "index.html"
    return FileResponse(str(index))


# Serve static assets from dist/assets/
from fastapi.staticfiles import StaticFiles
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="static-assets")

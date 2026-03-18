from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def create_app(brand_name: str, brands_root: Path) -> FastAPI:
    app = FastAPI(title=f"clip-video dashboard - {brand_name}")
    app.state.brand_name = brand_name
    app.state.brands_root = brands_root
    app.state.brand_path = brands_root / brand_name

    @app.get("/api/health")
    def health():
        return {"status": "ok", "brand": brand_name}

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app

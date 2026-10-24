"""Vercel entrypoint: the FastAPI preset picks up `app` from a root `app.py`.

Puts src/ first on the path so the package, and the data/ folder its config resolves
relative to it, load from this checkout rather than an installed copy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from property_intelligence_rag.server import app  # noqa: E402

__all__ = ["app"]

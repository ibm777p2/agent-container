"""Serve the agent over AG-UI (for CopilotKit or any AG-UI client), plus a plain JSON chat
endpoint and a direct property-research endpoint.

    uv run pi-serve   # http://127.0.0.1:8123/agent, /chat, /tools/property, /health, /healthz

`app` is also the Vercel entrypoint (see the root `app.py`).

When ``PI_AGENT_TOKEN`` is set, all cost-incurring endpoints require
``Authorization: Bearer <token>``. When ``PI_API_SECRET`` is set, /agent also requires the
``x-pi-secret`` header. Health checks remain public.

Browsers may call the service directly from any localhost port and from the origins in
``PI_WEB_ORIGINS`` (comma-separated; none by default).
"""

from __future__ import annotations

import hmac
import os
import secrets
from typing import Any

import uvicorn
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from property_intelligence_rag.graph import build_graph
from property_intelligence_rag.property_data import PropertyDataError, get_property_data

AGENT_TOKEN = os.getenv("PI_AGENT_TOKEN")
WEB_ORIGINS = [
    o.strip() for o in os.getenv("PI_WEB_ORIGINS", "").split(",") if o.strip()
]
LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def require_token(authorization: str | None = Header(default=None)) -> None:
    if AGENT_TOKEN and not secrets.compare_digest(
        authorization or "", f"Bearer {AGENT_TOKEN}"
    ):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


app = FastAPI(title="Property Intelligence RAG agent", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_ORIGINS,
    allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    max_age=600,
)
# When set, /agent only answers callers that hold the same secret (e.g. your own backend);
# this keeps scraper and model spend behind that caller's limits.
API_SECRET = os.getenv("PI_API_SECRET")


@app.middleware("http")
async def require_secret(request: Request, call_next):
    path = request.url.path
    if API_SECRET and request.method != "OPTIONS" and path.startswith("/agent") and not path.endswith("/health"):
        if not hmac.compare_digest(request.headers.get("x-pi-secret", "").encode(), API_SECRET.encode()):
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


add_langgraph_fastapi_endpoint(
    app,
    LangGraphAgent(
        name="default",
        description=(
            "Answers questions about the on-screen property or prototype, scraped listings, "
            "and live property comparables."
        ),
        # AG-UI resumes threads from checkpoints; conversations live only in this process.
        graph=build_graph(checkpointer=InMemorySaver()),
        emit_raw_events=False,
    ),
    "/agent",
    dependencies=[Depends(require_token)],
)


class PropertyRequest(BaseModel):
    address: str = Field(min_length=3, max_length=300)
    refresh: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tools/property", dependencies=[Depends(require_token)])
def property_tool(request: PropertyRequest) -> dict[str, Any]:
    try:
        return get_property_data(request.address, refresh=request.refresh)
    except (PropertyDataError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/chat", dependencies=[Depends(require_token)])
def chat(request: ChatRequest) -> dict[str, Any]:
    graph = build_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": request.message}]})
    message = result["messages"][-1]
    return {"answer": message.text, "message_type": message.type}


def main() -> None:
    uvicorn.run(
        app,
        host=os.getenv("PI_AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("PI_AGENT_PORT", os.getenv("PORT", "8123"))),
    )

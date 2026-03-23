#!/usr/bin/env python3
# http_facade.py — HTTP facade for your MCP index (same logic, same memory)
# Port: 5050, Auth: Bearer odyceo123 (or ?key=odyceo123)
# Endpoints:
#   GET  /health
#   GET  /resources/list?collection=raw
#   GET  /resources/read?uri=...&offset=0&limit=4000
#   POST /search { query, top_k, filters }
#   POST /prompt/retrieve { query, budget_tokens, filters }

import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse the in-memory index & parsers
import mcp_server  # ensures INDEX is built at import
from helpers.text_utils import token_estimate

# ---------------- Config ----------------
PORT = int(os.getenv("MCP_HTTP_PORT", "5050"))
# Accept both env override and default to your requested key
API_TOKEN = os.getenv("MCP_HTTP_TOKEN", "odyceo123")

app = FastAPI(
    title="MCP HTTP Facade",
    version="1.0.0",
    description="HTTP wrapper around mcp_server.INDEX (resources + search + prompt)",
)

# CORS (open for internal/demo use; tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Auth -------------------
async def auth_guard(
    authorization: Optional[str] = Header(default=None),
    key: Optional[str] = Query(default=None),
):
    """
    Allow either:
      - Authorization: Bearer <token>
      - ?key=<token>
    """
    token_ok = False
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
        token_ok = (bearer == API_TOKEN)
    elif key:
        token_ok = (key == API_TOKEN)

    if not token_ok:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# --------------- Schemas ----------------
class Filters(BaseModel):
    collections: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    filters: Optional[Filters] = None

class PromptRetrieveRequest(BaseModel):
    query: str
    budget_tokens: int = 800
    filters: Optional[Filters] = None

# --------------- Endpoints --------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "index_resources": len(mcp_server.INDEX.resources), "index_chunks": len(mcp_server.INDEX.chunks)}

@app.get("/resources/list", dependencies=[Depends(auth_guard)])
async def resources_list(collection: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    items = [r.asdict() for r in mcp_server.INDEX.list_resources(collection)]
    return {"resources": items, "next_page_token": None}

@app.get("/resources/read", dependencies=[Depends(auth_guard)])
async def resources_read(
    uri: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(4000, gt=0),
) -> Dict[str, Any]:
    try:
        content, cits, paging = mcp_server.INDEX.read_resource_text(uri, offset, limit)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"resource not found: {uri}")
    return {"uri": uri, "content": content, "citations": cits, "paging": paging}

@app.post("/search", dependencies=[Depends(auth_guard)])
async def search(req: SearchRequest) -> Dict[str, Any]:
    filt = {}
    if req.filters and req.filters.collections:
        filt["collections"] = req.filters.collections
    rows = mcp_server.INDEX.search(req.query, top_k=req.top_k, filters=filt)
    return {"results": rows}

@app.post("/prompt/retrieve", dependencies=[Depends(auth_guard)])
async def prompt_retrieve(req: PromptRetrieveRequest) -> Dict[str, Any]:
    filt = {}
    if req.filters and req.filters.collections:
        filt["collections"] = req.filters.collections

    # 1) lexical/semantic hybrid via INDEX.search
    #   - Grab a bit more than we need, then trim by token budget
    candidates = mcp_server.INDEX.search(req.query, top_k=max(12, req.budget_tokens // 200), filters=filt)

    blocks = []
    budget = int(req.budget_tokens)

    for r in candidates:
        try:
            # read whole locator chunk (paging happens inside)
            content, _, _ = mcp_server.INDEX.read_resource_text(r["uri"], 0, 6000)
        except KeyError:
            continue
        if not content.strip():
            continue

        est = token_estimate(content)
        if est <= budget:
            blocks.append({
                "type": "context",
                "source": r["uri"],
                "content": content,
                "confidence": float(r.get("score", 0.0)),
            })
            budget -= est
        else:
            # take a slice proportional to remaining budget
            # (simple safe slice; LLMs can still use it)
            ratio = max(0.25, min(1.0, budget / max(1, est)))
            take_chars = int(len(content) * ratio)
            snippet = content[:take_chars]
            if snippet.strip():
                blocks.append({
                    "type": "context",
                    "source": r["uri"],
                    "content": snippet,
                    "confidence": float(r.get("score", 0.0)),
                })
                budget = 0

        if budget <= 40:  # small safety tail
            break

    used = req.budget_tokens - budget
    return {"blocks": blocks, "usage": {"budget_tokens": req.budget_tokens, "used_tokens": used}}

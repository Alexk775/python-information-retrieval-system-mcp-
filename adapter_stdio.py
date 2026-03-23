#!/usr/bin/env python3
# adapter_stdio.py — pure stdio JSON-RPC adapter for your MCP (resources + prompt)
# Uses the same in-memory INDEX/services from mcp_server.py
# Transport: stdin/stdout pipes (no HTTP, no ports).

import sys, json, os, codecs

# Import your FastAPI module just to reuse the INDEX & helpers (it indexes on import)
try:
    import mcp_server  # ensure adapter_stdio.py is next to mcp_server.py
except ImportError as e:
    print(f"Failed to import mcp_server.py. Make sure it exists. Error: {e}", file=sys.stderr)
    sys.exit(1)


INDEX = mcp_server.INDEX

# Optional shared-secret for a simple handshake (set MCP_SHARED_KEY in env to enable)
SHARED = os.getenv("MCP_SHARED_KEY")


# --- Force UTF-8 pipes on Windows & sanitize undecodable bytes ---
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    # Python <3.7 fallback
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach(), errors="replace")
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach(), errors="replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach(), errors="replace")


def _u8(s: str) -> str:
    if not isinstance(s, str):
        return s
    # replace any "illegal" bytes
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize(obj):
    if isinstance(obj, str):
        return _u8(obj)
    if isinstance(obj, dict):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def respond(id_, result=None, error=None):
    """Write one JSON-RPC response line to stdout."""
    resp = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        resp["error"] = _sanitize(error)
    else:
        resp["result"] = _sanitize(result)
    try:
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception as e:
        # Handle cases where stdout might be closed
        print(f"Failed to write response: {e}", file=sys.stderr)


def capabilities():
    return {
        "protocol": "mcp/1.0",
        "resources": {"list": True, "read": True, "search": True, "metadata": False},
        "prompt": {"retrieve": True},
        "limits": {"max_read_bytes": 524288, "max_blocks": 16},
    }


def list_resources(params):
    coll = params.get("collection")
    items = [r.asdict() for r in INDEX.list_resources(coll)]
    return {"resources": items, "next_page_token": None}


def read_resource(params):
    uri = params["uri"]
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 4000))
    try:
        content, cits, paging = INDEX.read_resource_text(uri, offset, limit)
    except KeyError:
        raise KeyError(f"resource not found: {uri}")
    return {"uri": uri, "content": content, "citations": cits, "paging": paging}


def search(params):
    query = params["query"]
    top_k = int(params.get("top_k", 8))
    filters = params.get("filters") or {}
    return {"results": INDEX.search(query, top_k, filters)}


def token_estimate(text: str) -> int:
    # rough heuristic: ~4 chars/token
    return max(1, int(len(text) / 4))


def prompt_retrieve(params):
    query = params["query"]
    budget = int(params.get("budget_tokens", 1200))
    filters = params.get("filters") or {}

    main_results = INDEX.search(query, top_k=12, filters=filters)

    blocks = []

    def push(uri, snippet, kind="context", conf=0.9):
        nonlocal budget
        est = token_estimate(snippet)
        if est <= budget:
            blocks.append(
                {"type": kind, "source": uri, "content": snippet, "confidence": conf}
            )
            budget -= est

    for r in main_results:
        try:
            # We must read from the chunk, not the full resource
            content = INDEX.chunks.get(r["uri"])
            if not content:
                continue
        except KeyError:
            continue
        
        # The 'snippet' from search is already the start of the chunk
        snippet = r.get("snippet", "") 
        
        # FIX: Remove dependency on mcp_server.SEM_SNIPPET
        # We hard-code the value (600) from the mcp_server_v3 file.
        if len(snippet) >= (600 - 10): # Hard-coded SEM_SNIPPET value
             snippet = content[:2000] # Grab a larger chunk for context
        else:
             snippet = content # The whole chunk is small

        push(r["uri"], snippet, "context", r["score"])
        if budget <= 100:
            break

    used = int(params.get("budget_tokens", 1200)) - budget
    return {
        "blocks": blocks,
        "usage": {
            "budget_tokens": int(params.get("budget_tokens", 1200)),
            "used_tokens": used,
        },
    }


def main():
    # If MCP_SHARED_KEY is set, require a hello first
    authed = SHARED is None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            respond(None, error={"code": -32700, "message": f"Parse error: {e}"})
            continue

        id_ = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        try:
            if method == "mcp.hello":
                if SHARED and params.get("key") == SHARED:
                    authed = True
                    respond(id_, {"ok": True})
                elif not SHARED:
                     authed = True
                     respond(id_, {"ok": True})
                else:
                    respond(id_, error={"code": 401, "message": "unauthorized"})
                continue

            if not authed:
                respond(id_, error={"code": 401, "message": "unauthorized"})
                continue

            if method == "mcp.capabilities":
                respond(id_, capabilities())
            elif method == "resources.list":
                respond(id_, list_resources(params))
            elif method == "resources.read":
                respond(id_, read_resource(params))
            elif method == "resources.search":
                respond(id_, search(params))
            elif method == "prompt.retrieve":
                respond(id_, prompt_retrieve(params))
            else:
                respond(
                    id_,
                    error={"code": -32601, "message": f"Method not found: {method}"},
                )
        except KeyError as ke:
            respond(id_, error={"code": 404, "message": str(ke)})
        except Exception as e:
            # Print the full exception to stderr for debugging
            import traceback
            print(f"Error processing {method}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            respond(id_, error={"code": 500, "message": str(e)})


if __name__ == "__main__":
    # Importing mcp_server already indexed your files (on module import startup).
    print("Adapter starting, index loaded from mcp_server.", file=sys.stderr)
    main()


#!/usr/bin/env python3
# demo_email_local.py — MCP(stdio) -> Ollama (local) -> email draft (JSON)

import os, sys, json, uuid, subprocess, textwrap, re, requests

ADAPTER_CMD   = [sys.executable, "adapter_stdio.py"]
SHARED_KEY    = os.getenv("MCP_SHARED_KEY")                 # optional
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "phi3:mini")      # μικρό & ελαφρύ για 8GB RAM
CTX_BUDGET    = int(os.getenv("MCP_CTX_BUDGET", "400"))     # κράτα μικρό για ταχεία απόκριση

# ---------- stdio JSON-RPC ----------
def start_adapter():
    return subprocess.Popen(
        ADAPTER_CMD,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace"
    )

def rpc(p, method, params=None):
    req = {"jsonrpc":"2.0","id":str(uuid.uuid4()),"method":method,"params":params or {}}
    p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        err = (p.stderr.read() or "").strip()
        raise RuntimeError(f"No response for {method}. Stderr:\n{err}")
    resp = json.loads(line)
    if "error" in resp:
        raise RuntimeError(f"RPC error {method}: {resp['error']}")
    return resp["result"]

def collect_context(p, query, budget_tokens=CTX_BUDGET):
    if SHARED_KEY:
        rpc(p, "mcp.hello", {"key": SHARED_KEY})
    rpc(p, "mcp.capabilities")
    pr = rpc(p, "prompt.retrieve", {
        "query": query, "budget_tokens": budget_tokens, "filters": {}
    })
    blocks = pr.get("blocks", [])
    return "\n\n---\n\n".join(b.get("content","") for b in blocks if isinstance(b.get("content",""), str))

# ---------- prompt ----------
def build_prompts(context: str):
    system = (
        "You are an expert assistant that drafts crisp, polite business emails.\n"
        "Use the provided CONTEXT strictly for facts and phrasing cues if present.\n"
        "Return ONLY valid JSON with keys exactly: subject, body.\n"
        'Example: {"subject":"...","body":"..."}\n'
        "No extra text, no markdown, no code fences."
    )
    user = textwrap.dedent(f"""
        Write a concise, polite email requesting the availability of the venue during the
        dates of the Engineering World event. Include:

        - A clear subject line.
        - Greeting.
        - Dates as a placeholder: [DD–DD Month YYYY].
        - Expected headcount placeholder: [~N].
        - Setup needs: auditorium-style seating, projector/screen, basic PA/microphone.
        - Ask about catering options/policies (coffee breaks, light lunch).
        - Ask for available time slots and a quote (incl. AV/catering fees, taxes, payment terms).
        - Close with thanks and a call to action.
        - Keep it in English.

        CONTEXT:
        ---
        {context or "(no extra context)"}
        ---
    """).strip()
    return system, user

# ---------- Ollama call ----------
def call_ollama(system, user, model=OLLAMA_MODEL, url=OLLAMA_URL):
    r = requests.post(f"{url}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "options": {
            "num_ctx": 1024,      # καλό όριο για 8GB RAM
            "num_predict": 512,
            "temperature": 0.3
        },
        "stream": False
    }, timeout=180)
    r.raise_for_status()
    data = r.json()
    text = data.get("message", {}).get("content", "") or json.dumps(data)

    # 1) Κανονική προσπάθεια: καθαρό JSON
    try:
        obj = json.loads(text)
        return obj.get("subject", "(no subject)"), obj.get("body", text)
    except Exception:
        pass

    # 2) Rescue: πιάσε το πρώτο {...} μπλοκ
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj.get("subject","(no subject)"), obj.get("body", text)
        except Exception:
            pass

    # 3) Fallback: γύρνα το raw
    return "Request for venue availability during the Engineering World dates", text

def main():
    p = start_adapter()
    try:
        ctx_query = "venue availability request email for Engineering World dates"
        ctx = collect_context(p, ctx_query, budget_tokens=CTX_BUDGET)
        system, user = build_prompts(ctx)
        subject, body = call_ollama(system, user)

        print("Subject:", subject)
        print("\nBody:\n")
        print(body)

        with open("email_draft.txt","w",encoding="utf-8") as f:
            f.write(f"Subject: {subject}\n\n{body}\n")
        print("\n[Saved] email_draft.txt")
    finally:
        try: p.kill()
        except Exception: pass

if __name__ == "__main__":
    main()

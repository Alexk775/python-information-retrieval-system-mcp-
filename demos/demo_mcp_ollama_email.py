#!/usr/bin/env python3
# demo_mcp_ollama_email.py — MCP context -> local Ollama (qwen2:0.5b) -> email JSON

import os, json, textwrap, requests

MCP_URL   = os.environ.get("MCP_URL", "http://localhost:5050")
MCP_TOKEN = os.environ.get("MCP_HTTP_TOKEN", "odyceo123")
MODEL     = os.environ.get("OLLAMA_MODEL", "qwen2:0.5b")
OLLAMA    = os.environ.get("OLLAMA_URL", "http://localhost:11434")

def mcp_retrieve(query: str, budget_tokens: int = 800) -> str:
    r = requests.post(
        f"{MCP_URL}/prompt/retrieve",
        headers={"Authorization": f"Bearer {MCP_TOKEN}"},
        json={"query": query, "budget_tokens": budget_tokens, "filters": {}},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    blocks = data.get("blocks", [])
    return "\n\n---\n\n".join(b.get("content","") for b in blocks if b.get("content")).strip()

def call_ollama(system: str, user: str, model: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"num_ctx": 1024, "num_predict": 400}
    }
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    j = r.json()
    text = (j.get("message", {}).get("content") or j.get("response") or "").strip()
    try:
        obj = json.loads(text)
        return obj.get("subject","(no subject)"), obj.get("body", text)
    except Exception:
        return "Request for venue availability during the Engineering World dates", text

def build_prompts(context: str):
    system = (
        "You are a precise assistant that drafts crisp, polite business emails. "
        "Use ONLY facts/phrasings present in the CONTEXT. If a fact is missing, use a placeholder."
    )
    user = textwrap.dedent(f"""
        Task: Write a concise, polite email requesting the availability of a venue during the Engineering World event dates.

        Include:
        - Subject line.
        - Greeting.
        - Dates placeholder: [DD–DD Month YYYY].
        - Expected headcount placeholder: [~N].
        - Setup: auditorium-style seating, projector/screen, basic PA/microphone.
        - Ask about catering options/policies (coffee breaks, light lunch).
        - Ask for available time slots and a quote (AV/catering fees, taxes, payment terms).
        - English only.

        CONTEXT from my MCP files:
        ---
        {context}
        ---
        Return JSON with keys: subject, body.
    """).strip()
    return system, user

def main():
    query = "venue availability email Engineering World dates"
    print("[1/3] Pulling CONTEXT from MCP...")
    context = mcp_retrieve(query, budget_tokens=900)
    print(f"    Context size: {len(context)} chars")

    system, user = build_prompts(context)

    print(f"[2/3] Calling Ollama model: {MODEL} ...")
    subject, body = call_ollama(system, user, MODEL)

    print("\n[3/3] RESULT\n")
    print("Subject:", subject)
    print("\nBody:\n")
    print(body)

if __name__ == "__main__":
    main()

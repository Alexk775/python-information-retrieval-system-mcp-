#!/usr/bin/env python3
# demo_mcp_openrouter_email_requests.py
# This version is a "freer" test to evaluate the *MCP's retrieval quality*.
# It removes strict JSON formatting to let the LLM use the context.

import os, sys, json, uuid, subprocess, textwrap, requests
import time
import re

ADAPTER_CMD = [sys.executable, "adapter_stdio.py"]
SHARED_KEY = os.environ.get("MCP_SHARED_KEY") # optional
OPENROUTER_KEY = (
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
)
BASE_URL = "https://openrouter.ai/api/v1"

# New, working list from your other file
MODEL_CANDIDATES = [
    "meta-llama/llama-3.3-8b-instruct:free",
    "zhipu/glm-4.5-air:free",
    "minimax/minimax-m2:free",
]


if not OPENROUTER_KEY:
    print("ERROR: set OPENROUTER_API_KEY in your env", file=sys.stderr)
    sys.exit(1)

# ---------- MCP stdio JSON-RPC ----------
def send_rpc(p, method, params=None, id_=None):
    if id_ is None:
        id_ = str(uuid.uuid4())
    req = {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}
    p.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
    p.stdin.flush()
    line = p.stdout.readline().decode("utf-8", errors="replace")
    if not line:
        raise RuntimeError("No response from adapter.")
    resp = json.loads(line)
    if "error" in resp:
        raise RuntimeError(f"RPC error {method}: {resp['error']}")
    return resp["result"]


def collect_context_blocks(p, query, budget_tokens=2000): # Increased budget
    if SHARED_KEY:
        send_rpc(p, "mcp.hello", {"key": SHARED_KEY})
    send_rpc(p, "mcp.capabilities")
    pr = send_rpc(
        p,
        "prompt.retrieve",
        {"query": query, "budget_tokens": budget_tokens, "filters": {}},
    )
    blocks = pr.get("blocks", [])

    cleaned = []
    # Get the best context
    for b in blocks[:7]: # Focus on the top 7
        src = b.get("source", "")
        txt = (b.get("content", "") or "").strip()
        if not txt:
            continue
        snippet = txt[:1000] # Use a generous snippet
        cleaned.append({"source": src, "snippet": snippet})
    return cleaned


# ---------- Prompt (Simplified) ----------
def build_prompt(blocks, task_query):
    """Builds a simpler, non-JSON prompt."""

    if not blocks:
        ctx = "No context retrieved from MCP."
    else:
        ctx_lines = []
        for i, b in enumerate(blocks, 1):
            ctx_lines.append(
                f'--- Context Snippet {i} (Source: {b["source"]}) ---\n{b["snippet"]}'
            )
        ctx = "\n\n".join(ctx_lines)

    system = textwrap.dedent("""
    You are a helpful assistant. Your job is to complete the user's task based *only* on the provided context snippets.
    Do not make up information.
    If the context provides specific details (like names, dates, or numbers), you MUST use them.
    If the context does not provide a detail, use a placeholder like [DETAIL_MISSING].
    """).strip()

    user = textwrap.dedent(f"""
    CONTEXT SNIPPETS:
    {ctx}

    TASK:
    Based *only* on the context above, please perform the following task:
    "{task_query}"
    """).strip()

    return system, user


# ---------- OpenRouter (Simplified) ----------
def call_openrouter(system, user):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        # For server-side calls "HTTP-Referer" is optional; you can keep it or remove it.
        "HTTP-Referer": "http://localhost",
        "X-Title": "mcp-email-demo-freer",
    }

    base_payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    max_attempts_per_model = 3
    backoff_seconds = [2, 5, 10, 15]
    last_err = None

    for model in MODEL_CANDIDATES:
        for attempt in range(max_attempts_per_model):
            try:
                payload = {"model": model, **base_payload}
                r = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                # queued / rate-limited
                if r.status_code in (202, 429):
                    wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                    time.sleep(wait)
                    continue

                # not found -> try next model
                if r.status_code == 404:
                    last_err = RuntimeError(f"Model '{model}' returned 404 (not found).")
                    print(f"[WARN] {last_err} Trying next model...")
                    break  # stop retrying this model

                # payment required -> stop now (account/credits issue)
                if r.status_code == 402:
                    detail = (r.text or "")[:300]
                    raise RuntimeError(
                        "402 Payment Required from OpenRouter. "
                        "Likely no free credits / unverified / negative balance. "
                        "Verify phone or add a small credit, then rotate the API key. "
                        f"Details: {detail}"
                    )

                # forbidden -> stop now (key/referrer/permissions)
                if r.status_code == 403:
                    detail = (r.text or "")[:300]
                    raise RuntimeError(
                        "403 Forbidden from OpenRouter. "
                        "Likely invalid/disabled key, blocked referrer, or model not allowed. "
                        f"Details: {detail}"
                    )

                # other http errors
                r.raise_for_status()

                data = r.json()
                text = (
                    (data.get("choices", [{}])[0].get("message", {}) or {})
                    .get("content", "")
                    or ""
                ).strip()

                if text:
                    print(f"\n[LLM ({model}) SUCCEEDED]\n")
                    return text

                # Empty completion -> treat as failure & retry
                last_err = RuntimeError(
                    f"Empty completion from model '{model}'. Raw: {str(data)[:300]}"
                )
                print("[WARN]", last_err)
                wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                time.sleep(wait)

            except Exception as e:
                last_err = e
                wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                time.sleep(wait)
                continue

        time.sleep(2)

    raise RuntimeError(f"All models failed. Last error: {last_err}")




# ---------- Main ----------
def main(mcp_query, llm_task):
    p = subprocess.Popen(
        ADAPTER_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # 1. Retrieve context from MCP
        print(f"--- Querying MCP for: '{mcp_query}' ---")
        blocks = collect_context_blocks(p, mcp_query, budget_tokens=2000)

        if not blocks:
            print("!!! MCP returned no context blocks. Aborting. !!!")
            return

        print("\n=== MCP RETRIEVED (first 120 chars) ===")
        for i, b in enumerate(blocks, 1):
            print(f"[{i}] {b['source']} => {b['snippet'][:125].replace(os.linesep, ' ')}")
        print(f"\n--- Sending to LLM for task: '{llm_task}' ---")

        # 2. Build a simple prompt
        system, user = build_prompt(blocks, llm_task)

        # 3. Get the LLM to generate text
        llm_output = call_openrouter(system, user)

        print("=== FINAL LLM OUTPUT ===")
        print(llm_output)

    except Exception as e:
        print(f"\n*** TEST FAILED ***\n{e}")
    finally:
        try:
            p.kill()
        except Exception:
            pass


if __name__ == "__main__":
    # --- THIS IS WHERE YOU TEST ---
    # Now you can test different queries and tasks to see your MCP's value.

    # TEST 1: The original email request (more open-ended)
    mcp_query_1 = "Information for Engineering the World event: venues, dates, attendance"
    llm_task_1 = "Draft a short, professional email to a potential venue. Ask about availability for the 'Engineering the World' event. Make sure to mention the potential venues, dates, and expected attendance number found in the context."
    
    # TEST 2: A different task (summarization)
    mcp_query_2 = "What is the APD program?"
    llm_task_2 = "Summarize the 'Annual Project Development (APD) Program' in one paragraph."

    # TEST 3: A specific question
    mcp_query_3 = "Who is on the ODYCEO core team?"
    llm_task_3 = "List the names of the ODYCEO core team members mentioned in the context."


    print("========== RUNNING TEST 1 (Email) ==========")
    main(mcp_query_1, llm_task_1)
    
    print("\n\n========== RUNNING TEST 2 (Summarize APD) ==========")
    main(mcp_query_2, llm_task_2)

    print("\n\n========== RUNNING TEST 3 (Core Team) ==========")
    main(mcp_query_3, llm_task_3)

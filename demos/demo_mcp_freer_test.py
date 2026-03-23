#!/usr/bin/env python3
# demo_mcp_freer_test.py
# This version is a "freer" test to evaluate the *MCP's retrieval quality*.
# It removes strict JSON formatting to let the LLM use the context.

import os, sys, json, uuid, subprocess, textwrap, requests
import time
import re

ADAPTER_CMD = [sys.executable, "adapter_stdio.py"]
SHARED_KEY = os.environ.get("MCP_SHARED_KEY")  # optional
OPENROTER_KEY = (
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
)
BASE_URL = "https://openrouter.ai/api/v1"

# --- FIX: More reliable list of free models to avoid 404 ---
MODEL_CANDIDATES = [
    "meta-llama/llama-3-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-7b-it:free",
    "nousresearch/nous-hermes-2-mistral-7b-dpo:free",
]

if not OPENROTER_KEY:
    print("ERROR: set OPENROUTER_API_KEY in your env", file=sys.stderr)
    sys.exit(1)

# ---------- MCP stdio JSON-RPC ----------
def send_rpc(p, method, params=None, id_=None):
    if id_ is None:
        id_ = str(uuid.uuid4())
    req = {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}
    try:
        p.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        p.stdin.flush()
    except BrokenPipeError:
        raise RuntimeError("Adapter process terminated unexpectedly. (BrokenPipeError)")
    
    line = p.stdout.readline().decode("utf-8", errors="replace")
    if not line:
        stderr_output = p.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"No response from adapter. It may have crashed. Stderr: {stderr_output}")
    
    try:
        resp = json.loads(line)
    except json.JSONDecodeError as e:
        stderr_output = p.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to decode JSON response: {e}. Raw response: '{line}'. Stderr: {stderr_output}")

    if "error" in resp:
        raise RuntimeError(f"RPC error {method}: {resp['error']}")
    return resp["result"]


def collect_context_blocks(p, query, budget_tokens=1400):
    if SHARED_KEY:
        send_rpc(p, "mcp.hello", {"key": SHARED_KEY})
    send_rpc(p, "mcp.capabilities")
    pr = send_rpc(
        p, "prompt.retrieve", {"query": query, "budget_tokens": budget_tokens, "filters": {}}
    )
    blocks = pr.get("blocks", [])

    cleaned = []
    # keep up to 8 blocks
    for b in blocks[:8]:
        src = b.get("source", "")
        txt = (b.get("content", "") or "").strip()
        if not txt:
            continue
        # Use the snippet *returned* from the search, which is already a good size
        snippet = b.get("snippet", txt[:800])
        cleaned.append({"source": src, "snippet": snippet})
    return cleaned


# ---------- Prompt ----------
def build_prompt(query, task, blocks):
    ctx = "\n\n".join(
        f"[{i+1}] Source: {x['source']}\nContent: {x['content']}" # <-- Use the 'content' key
        for i, x in enumerate(blocks)
    )

    system = """
You are a professional assistant. Your task is to answer the user's query based *only* on the provided context snippets.
Do not use any outside knowledge. If the answer is not in the snippets, say so.
Be concise and directly answer the question.
""".strip()

    user = textwrap.dedent(
        f"""
    CONTEXT SNIPPETS:
    ---
    {ctx if ctx else "No context provided."}
    ---

    QUERY:
    {query}

    TASK:
    {task}
    """
    ).strip()

    return system, user


# ---------- OpenRouter ----------
def call_openrouter(system, user):
    headers = {
        "Authorization": f"Bearer {OPENROTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Recommended by OpenRouter
        "X-Title": "mcp-freer-test",
    }

    base_payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }

    max_attempts_per_model = 2
    backoff_seconds = [2, 5]
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

                if r.status_code == 202:  # Queued
                    time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                    continue
                
                if r.status_code == 429:  # Rate limit
                    time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                    continue

                r.raise_for_status() # Raise HTTPError for bad responses (4xx, 5xx)

                data = r.json()
                text = (
                    (data.get("choices", [{}])[0].get("message", {}) or {})
                    .get("content", "")
                    or ""
                ).strip()
                
                if not text:
                    raise ValueError("Empty response from model")

                # Clean up common model prefixes/suffixes
                text = re.sub(r"<s>\s*\[OUT\]\s*", "", text)
                text = re.sub(r"\[/OUT\]\s*$", "", text)

                print(f"[LLM ({model}) SUCCEEDED]")
                return text, None

            except Exception as e:
                last_err = e
                print(f"[LLM ({model}) attempt {attempt+1} failed: {e}]", file=sys.stderr)
                time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                continue
        
        time.sleep(1) # Small pause before trying next model

    return None, last_err

# ---------- Main ----------
def run_test(p, test_name, query, task):
    print(f"\n\n========== RUNNING TEST {test_name} ==========")
    print(f"--- Querying MCP for: '{query}' ---\n")
    
    try:
        blocks = send_rpc(p, "prompt.retrieve", {
            "query": query, "budget_tokens": 2000, "filters": {}
        }).get("blocks", [])

        print("=== MCP RETRIEVED (first 120 chars) ===")
        if not blocks:
            print("(No blocks retrieved)")
        for i, b in enumerate(blocks[:5], 1): # Show top 5
            snippet_preview = b.get("content", "")[:120].replace("\n", " ")
            print(f"[{i}] {b.get('source')} => {snippet_preview}")
        print()

        system, user = build_prompt(query, task, blocks)

        print(f"--- Sending to LLM for task: '{task}' ---\n")
        final_text, err = call_openrouter(system, user)

        if err:
            print(f"\n*** TEST FAILED ***")
            print(f"All models failed. Last error: {err}")
            return

        print("\n=== FINAL LLM OUTPUT ===")
        print(final_text or "(No output from LLM)")

    except Exception as e:
        print(f"\n*** TEST FAILED ***")
        print(f"An error occurred: {e}")
        # Try to read stderr from the crashed process
        try:
            stderr_output = p.stderr.read().decode("utf-8", errors="replace")
            if stderr_output:
                print("\n--- Adapter Stderr ---")
                print(stderr_output)
        except Exception:
            pass # Process might already be dead


def main():
    p = None
    try:
        # Start the adapter process
        p = subprocess.Popen(
            ADAPTER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr to see crashes
        )
        
        # --- TEST 1 ---
        run_test(
            p,
            "1 (Email)",
            "Information for Engineering the World event: venues, dates, attendance",
            "Draft a short, professional email to a potential venue. Ask about availability for the 'Engineering the World' event. Make sure to mention the potential venues, dates, and expected attendance number found in the context."
        )

        # --- TEST 2 ---
        run_test(
            p,
            "2 (Summarize APD)",
            "What is the APD program?",
            "Summarize the 'Annual Project Development (APD) Program' in one paragraph."
        )

        # --- TEST 3 ---
        run_test(
            p,
            "3 (Core Team)",
            "Who is on the ODYCEO core team?",
            "List the names of the ODYCEO core team members mentioned in the context."
        )

    finally:
        if p:
            try:
                p.stdin.close() # Close stdin to signal EOF
                p.terminate() # Terminate the process
                p.wait(timeout=5) # Wait for it to exit
            except Exception as e:
                print(f"Error while cleaning up adapter process: {e}", file=sys.stderr)
                try:
                    p.kill() # Force kill if terminate fails
                except Exception:
                    pass

if __name__ == "__main__":
    main()

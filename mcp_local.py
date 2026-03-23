# mcp_local.py
from typing import List, Dict, Optional
import mcp_server
from helpers.text_utils import token_estimate

INDEX = mcp_server.INDEX

def mcp_retrieve(query: str, budget: int = 1200, collections: Optional[List[str]] = None) -> List[Dict]:
    """Επιστρέφει blocks [{'source','content','confidence'}] από το local INDEX."""
    filt = {}
    if collections:
        filt["collections"] = collections
    candidates = INDEX.search(query, top_k=max(12, budget // 200), filters=filt)

    blocks = []
    rem = int(budget)
    for r in candidates:
        try:
            content, _, _ = INDEX.read_resource_text(r["uri"], 0, 6000)
        except KeyError:
            continue
        if not content.strip():
            continue
        est = token_estimate(content)
        if est <= rem:
            blocks.append({"source": r["uri"], "content": content, "confidence": float(r.get("score", 0.0))})
            rem -= est
        else:
            take = int(len(content) * max(0.25, min(1.0, rem / max(1, est))))
            snip = content[:take]
            if snip.strip():
                blocks.append({"source": r["uri"], "content": snip, "confidence": float(r.get("score", 0.0))})
                rem = 0
        if rem <= 40:
            break
    return blocks

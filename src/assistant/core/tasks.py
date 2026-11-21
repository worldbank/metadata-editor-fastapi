"""One-shot task helpers (no agent loop)."""

from __future__ import annotations

import json
import re


def parse_suggestion(text: str) -> dict[str, str]:
    """Extract {suggestion, reason} from a model reply, tolerating code fences."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).removesuffix("```").strip()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except (json.JSONDecodeError, TypeError):
            obj = {}
    suggestion = str(obj.get("suggestion") or obj.get("value") or "").strip()
    reason = str(obj.get("reason") or "").strip()
    if not suggestion and raw and not reason:
        suggestion = raw
    return {"suggestion": suggestion, "reason": reason}

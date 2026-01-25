import json
import re
from typing import Any, Dict, Optional

JSON_RE = re.compile(r"\{[\s\S]*\}")
WRAP_RE = re.compile(r"<START>\s*(.*?)\s*<END>", re.DOTALL)


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if match:
            return json.loads(match.group(0))
    raise ValueError("Model did not return valid JSON")


def extract_wrapped_text(text: str) -> Optional[str]:
    match = WRAP_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score

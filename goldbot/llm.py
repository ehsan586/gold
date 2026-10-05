"""OPTIONAL advisory LLM layer. DISABLED by default. It can never send orders: its output is
validated text that is only stored next to the decision in the journal.
Key comes from the environment (ANTHROPIC_API_KEY); nothing is hardcoded."""
from __future__ import annotations

import json
import os
import time
import urllib.request

ALLOWED = {"AGREE", "DISAGREE", "UNSURE"}


def validate(text: str) -> dict:
    d = json.loads(text)
    if not isinstance(d, dict) or d.get("verdict") not in ALLOWED:
        raise ValueError("bad verdict")
    note = str(d.get("note", ""))[:300]
    return {"verdict": d["verdict"], "note": note}


class LLMAdvisor:
    def __init__(self, transport=None, timeout: float = 10.0, retries: int = 1):
        self.enabled = os.environ.get("GOLDBOT_LLM_ENABLED", "false").lower() == "true"
        self.transport, self.timeout, self.retries = transport or self._anthropic, timeout, retries

    def _anthropic(self, prompt: str) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        body = json.dumps({"model": os.environ.get("GOLDBOT_LLM_MODEL", "claude-sonnet-5-5"), "max_tokens": 200,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["content"][0]["text"]

    def review(self, summary: dict) -> dict | None:
        if not self.enabled:
            return None
        prompt = ("You review a proposed XAUUSD trade for sanity. Reply ONLY with JSON "
                  '{"verdict":"AGREE|DISAGREE|UNSURE","note":"<=200 chars"}.\n' + json.dumps(summary)[:3000])
        for attempt in range(self.retries + 1):
            try:
                return validate(self.transport(prompt))
            except Exception:  # noqa: BLE001 - LLM failure must never affect trading
                time.sleep(0.5 * (attempt + 1))
        return None

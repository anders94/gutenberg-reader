"""httpx wrapper for the Ollama /api/chat endpoint."""

from __future__ import annotations
import json
import time
from typing import Any

import httpx


class OllamaError(Exception):
    pass


def _fix_json_strings(content: str) -> str:
    """Fix common LLM JSON output errors inside string values:

    1. Unescaped control characters (newline, tab, carriage return)
    2. Unescaped dialogue quotes: ``"text": ""From...`` → ``"text": "\"From...``
       (the model uses the dialogue opening quote unescaped as the first char)
    """
    result = []
    in_string = False
    just_opened = False  # True for one char after the opening "
    i = 0
    while i < len(content):
        c = content[i]
        if c == "\\" and in_string:
            # Valid escape sequence — pass through both chars unchanged
            just_opened = False
            result.append(c)
            i += 1
            if i < len(content):
                result.append(content[i])
            i += 1
            continue
        if c == '"':
            if not in_string:
                in_string = True
                just_opened = True
                result.append(c)
                i += 1
                continue
            else:
                nxt = content[i + 1] if i + 1 < len(content) else ""
                nxt2 = content[i + 2] if i + 2 < len(content) else ""
                # Pattern 1: just_opened → "" at start of string value
                # e.g. "text": ""Hello..." — second " is unescaped dialogue opening quote
                if just_opened and nxt and nxt not in (",", "}", "]", "\n", "\r", " ", "\t"):
                    result.append('\\"')
                    just_opened = False
                    i += 1
                    continue
                # Pattern 2: any "" sequence inside a string — first " is always an embedded quote.
                # Covers end-of-value ("...world."",) and mid-string ("...uncle. ""Your great men...)
                if not just_opened and nxt == '"':
                    result.append('\\"')
                    just_opened = False
                    i += 1
                    continue
                # Pattern 3: " immediately followed by a non-structural character = mid-string embedded quote
                # In valid JSON, a closing " can't be immediately followed by a word/punctuation char —
                # only by comma, brace, bracket, colon, whitespace, or another quote.
                _STRUCTURAL = {",", "}", "]", ":", '"', "\n", "\r", "\t", " ", ""}
                if not just_opened and nxt not in _STRUCTURAL:
                    result.append('\\"')
                    just_opened = False
                    i += 1
                    continue
                # Normal close
                in_string = False
                just_opened = False
        elif in_string:
            just_opened = False
            if c == "\n":
                result.append("\\n")
                i += 1
                continue
            elif c == "\r":
                result.append("\\r")
                i += 1
                continue
            elif c == "\t":
                result.append("\\t")
                i += 1
                continue
        else:
            just_opened = False
        result.append(c)
        i += 1
    return "".join(result)


def _strip_trailing_commas(content: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON error)."""
    import re
    return re.sub(r",(\s*[}\]])", r"\1", content)


def _strip_markdown_fences(content: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        # Remove closing fence
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3].rstrip()
    return content


def _recover_partial_json(content: str) -> Any | None:
    """Try to recover usable data from a truncated JSON response.

    Attempts progressively shorter truncations of the content, closing open
    arrays/objects, until something parses. Returns the result or None.
    """
    # Try closing open structures at each candidate truncation point
    # Work backwards through the string looking for the last complete object
    for cut in range(len(content), max(len(content) - 2000, 0), -1):
        candidate = content[:cut]
        # Count unclosed brackets to determine what suffix to add
        depth_curly = candidate.count("{") - candidate.count("}")
        depth_square = candidate.count("[") - candidate.count("]")
        if depth_curly < 0 or depth_square < 0:
            continue
        suffix = "]" * depth_square + "}" * depth_curly
        try:
            result = json.loads(candidate + suffix)
            # Only return if we got something with at least one segment
            if isinstance(result, dict) and result.get("segments"):
                return result
        except json.JSONDecodeError:
            continue
    return None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health_check(self, model: str) -> None:
        """Verify Ollama is running and the model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {e}") from e

        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        # Accept exact match or prefix match (model names may include tag)
        if not any(m == model or m.startswith(model + ":") or model.startswith(m.split(":")[0]) for m in models):
            suggestion = f"  Try: gutenberg-reader ... --model {models[0]}" if models else ""
            raise OllamaError(
                f"Model '{model}' not found in Ollama.\n"
                f"  Available: {', '.join(models)}\n"
                + suggestion
            )

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
        require_json: bool = True,
    ) -> str:
        """Send a chat request and return the assistant's message content."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # disable chain-of-thought for thinking models (e.g. gemma4)
            "options": {
                "temperature": temperature,
                "num_predict": -1,   # unlimited output tokens
                "num_ctx": 16384,    # ensure large enough context window
            },
        }
        if require_json:
            payload["format"] = "json"

        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama /api/chat failed: {e}") from e

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
    ) -> Any:
        """Send a chat request and parse the JSON response.

        Strips markdown code fences if present. If truncated, attempts partial recovery.
        """
        content = self.chat(model, messages, temperature=temperature, require_json=True)
        content = _strip_markdown_fences(content)
        content = _fix_json_strings(content)
        content = _strip_trailing_commas(content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            recovered = _recover_partial_json(content)
            if recovered is not None:
                return recovered
            pos = e.pos
            snippet = content[max(0, pos - 100):pos + 100]
            raise OllamaError(
                f"Ollama returned invalid JSON: {e}\n"
                f"Content around pos {pos}: {snippet!r}"
            ) from e

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

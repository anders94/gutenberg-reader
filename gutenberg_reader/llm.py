"""httpx client for OpenAI-compatible chat APIs (vLLM, llama.cpp server, LM Studio, ...)."""

from __future__ import annotations
import json
from typing import Any
from urllib.parse import urlparse

import httpx


class LLMError(Exception):
    pass


class LLMStalled(LLMError):
    """The server cut the response at max_tokens.

    Two things do this: an answer that is genuinely long, and a model stalled
    under guided decoding — wanting a token the grammar masks, it emits the
    whitespace the grammar allows, forever. The second is what an unbounded
    request turns into a 64,000-token, nine-minute failure; the cap makes it
    a short one, and the retry can then change the sampling to break it.
    """


# Completion tokens per request. Well above any real answer (a critic window
# with thinking runs ~2k, the structure pass measured 5k on a large model) and
# far below the 65k context a stalled request otherwise consumes.
DEFAULT_MAX_TOKENS = 16384

# Applied on the retry after a stall. Measured on the roster review that
# stalled: 1.15 turns the whitespace loop into a complete answer in 3 seconds;
# temperature 0.7 alone does not.
STALL_RETRY_REPETITION_PENALTY = 1.15


def _strip_markdown_fences(content: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3].rstrip()
    return content


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Works with any server exposing /v1/models and /v1/chat/completions.
    When a JSON schema is supplied, it is passed as a structured-output
    constraint (response_format json_schema), which vLLM enforces via
    guided decoding — the response is guaranteed to parse and validate.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        template_kwargs: dict | None = None,
    ):
        base_url = base_url.rstrip("/")
        # Accept bare host URLs like http://host:8000 — the OpenAI API lives under /v1
        if not urlparse(base_url).path.strip("/"):
            base_url += "/v1"
        self.base_url = base_url
        self.timeout = timeout
        # Passed through to the server's chat template. Servers ignore keys their
        # template does not declare, so this is safe to set per endpoint.
        self.template_kwargs = template_kwargs or {}
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def health_check(self, model: str | None = None) -> str:
        """Verify the server is up; return the resolved model name.

        If model is empty/None, auto-select the first model the server offers
        (the common single-model vLLM case).
        """
        try:
            resp = self._client.get(f"{self.base_url}/models")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(
                f"Cannot reach OpenAI-compatible API at {self.base_url}: {e}\n"
                f"  Is vLLM running? e.g.: vllm serve <model> ..."
            ) from e

        available = [m["id"] for m in resp.json().get("data", [])]
        if not model:
            if not available:
                raise LLMError(f"No models available at {self.base_url}")
            return available[0]

        if not any(m == model or m.startswith(model) or model.startswith(m) for m in available):
            raise LLMError(
                f"Model '{model}' not found at {self.base_url}.\n"
                f"  Available: {', '.join(available) or '(none)'}"
            )
        return model

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
        schema: dict | None = None,
        require_json: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sampling: dict | None = None,
    ) -> str:
        """Send a chat request and return the assistant's message content.

        sampling: extra sampling parameters passed through to the server
        (repetition_penalty and the like). Raises LLMStalled when the server
        cut the answer at max_tokens.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **(sampling or {}),
        }
        if self.template_kwargs:
            payload["chat_template_kwargs"] = self.template_kwargs
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        elif require_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500]
            raise LLMError(f"chat/completions failed: {e}\n  {detail}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"chat/completions failed: {e}") from e

        data = resp.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape: {json.dumps(data)[:500]}") from e
        if choice.get("finish_reason") == "length":
            used = (data.get("usage") or {}).get("completion_tokens", "?")
            raise LLMStalled(
                f"response cut at max_tokens={max_tokens} ({used} completion "
                f"tokens; a model stalled under guided decoding emits whitespace "
                f"until it hits the cap). Content so far: {content[:200]!r}"
            )
        return content

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.1,
        schema: dict | None = None,
        **kw,
    ) -> Any:
        """Send a chat request and parse the JSON response."""
        content = self.chat(model, messages, temperature=temperature, schema=schema, **kw)
        content = _strip_markdown_fences(content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            snippet = content[max(0, e.pos - 100):e.pos + 100]
            raise LLMError(
                f"Model returned invalid JSON: {e}\n"
                f"Content around pos {e.pos}: {snippet!r}"
            ) from e

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# A window that fails is worth retrying; a window that fails silently is not.
# Only the attribution pass ever retried, and only it noticed — stage 02, the
# critic and character discovery each swallowed the error and lost that window.
DEFAULT_RETRIES = 3


def call_json_with_retries(
    client: "LLMClient | LLMRouter",
    model: str,
    messages: list[dict],
    schema: dict | None = None,
    retries: int = DEFAULT_RETRIES,
    what: str = "LLM call",
    console=None,
    temperature: float = 0.1,
) -> Any | None:
    """chat_json with retries. Returns None once every attempt has failed.

    Returning None rather than raising is deliberate: one bad window should cost
    its own text, not the chapter. But the caller must then *record* the loss —
    a skipped critic window that still counts toward a quality score is how a
    review reports 0.94 on work it never looked at.
    """
    last: Exception | None = None
    sampling: dict | None = None
    for attempt in range(1, retries + 1):
        try:
            return client.chat_json(
                model, messages, schema=schema, temperature=temperature,
                sampling=sampling)
        except LLMError as e:
            last = e
            if isinstance(e, LLMStalled):
                # Same prompt, same greedy answer, same stall. Change what the
                # model is allowed to prefer rather than asking again.
                sampling = {"repetition_penalty": STALL_RETRY_REPETITION_PENALTY}
            if console is not None:
                console.print(
                    f"  [red]{what} failed (attempt {attempt}/{retries}): {e}[/red]"
                )
    if console is not None:
        console.print(f"  [red]{what} gave up after {retries} attempts: {last}[/red]")
    return None


class LLMRouter:
    """Dispatches by model name to whichever endpoint serves that model.

    The pipeline runs bulk work on one server and the judgment calls — verify,
    tie-break, critic, structure — on another, usually a much larger model on
    different hardware. Every call site already passes a model name, so routing
    on that name means none of them have to learn about endpoints.
    """

    def __init__(self) -> None:
        self._by_model: dict[str, LLMClient] = {}

    def register(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "",
        timeout: float = 300.0,
        template_kwargs: dict | None = None,
    ) -> str:
        """Health-check an endpoint, resolve its model name, and route to it.

        Returns the resolved name, which is what callers pass back in.
        """
        client = LLMClient(base_url=base_url, api_key=api_key, timeout=timeout,
                           template_kwargs=template_kwargs)
        resolved = client.health_check(model or None)
        self._by_model[resolved] = client
        return resolved

    def endpoint_for(self, model: str) -> str:
        return self._by_model[model].base_url

    def _client(self, model: str) -> "LLMClient":
        try:
            return self._by_model[model]
        except KeyError:
            raise LLMError(
                f"No endpoint registered for model {model!r}. "
                f"Registered: {', '.join(sorted(self._by_model)) or '(none)'}"
            ) from None

    def chat(self, model: str, messages: list[dict], **kw) -> str:
        return self._client(model).chat(model, messages, **kw)

    def chat_json(self, model: str, messages: list[dict], **kw) -> Any:
        return self._client(model).chat_json(model, messages, **kw)

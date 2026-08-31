"""Endpoint routing and shared retry.

The pipeline runs bulk work on one server and the judgment passes on another.
Two things had to be true before that was possible: each role has to be
health-checked against the server that actually serves it, and a call that fails
has to be retried rather than silently dropped.
"""

from __future__ import annotations

import pytest

from gutenberg_reader import llm
from gutenberg_reader.config import Config
from gutenberg_reader.llm import LLMError, LLMRouter, call_json_with_retries


class _StubClient:
    """Stands in for LLMClient without a server."""

    def __init__(self, base_url, api_key="EMPTY", timeout=300.0, template_kwargs=None):
        self.base_url = base_url
        self.timeout = timeout
        self.template_kwargs = template_kwargs or {}
        self.calls: list[str] = []

    def health_check(self, model=None):
        # Each fake endpoint serves exactly one model, named after its host.
        served = "big-model" if "mac" in self.base_url else "small-model"
        if model and model != served:
            raise LLMError(f"Model {model!r} not found at {self.base_url}")
        return served

    def chat_json(self, model, messages, **kw):
        self.calls.append(model)
        return {"model": model, "endpoint": self.base_url}


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(llm, "LLMClient", _StubClient)


def test_router_dispatches_by_model(stub):
    r = LLMRouter()
    small = r.register("http://gpu:8000/v1")
    big = r.register("http://mac:8000/v1")

    assert (small, big) == ("small-model", "big-model")
    assert r.chat_json(small, [])["endpoint"] == "http://gpu:8000/v1"
    assert r.chat_json(big, [])["endpoint"] == "http://mac:8000/v1"


def test_validator_is_checked_against_its_own_endpoint(stub):
    """The blocker this replaces: validation_model was health-checked against
    the *processing* endpoint, so pointing --validator at a model only the
    second box serves killed the run at startup."""
    r = LLMRouter()
    r.register("http://gpu:8000/v1")
    assert r.register("http://mac:8000/v1", model="big-model") == "big-model"

    # And the same model name against the wrong endpoint still fails loudly.
    with pytest.raises(LLMError, match="not found"):
        r.register("http://gpu:8000/v1", model="big-model")


def test_unregistered_model_names_what_is_registered(stub):
    r = LLMRouter()
    r.register("http://gpu:8000/v1")
    with pytest.raises(LLMError, match="small-model"):
        r.chat_json("nonexistent", [])


def test_per_role_timeouts_reach_the_client(stub):
    """A large model on other hardware takes far longer per call than a 4B on a
    4090; a structure pass measured against PG 6400 hit the old fixed 300s."""
    r = LLMRouter()
    r.register("http://gpu:8000/v1", timeout=300.0)
    r.register("http://mac:8000/v1", timeout=1800.0)
    assert r._client("small-model").timeout == 300.0
    assert r._client("big-model").timeout == 1800.0


# ── Shared retry ─────────────────────────────────────────────────────────────

class _Flaky:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.attempts = 0

    def chat_json(self, model, messages, **kw):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise LLMError("boom")
        return {"ok": True}


def test_retry_succeeds_after_transient_failures():
    c = _Flaky(fail_times=2)
    assert call_json_with_retries(c, "m", [], retries=3) == {"ok": True}
    assert c.attempts == 3


def test_retry_returns_none_when_exhausted():
    """None rather than an exception: one bad window costs its own text, not the
    chapter. The caller is then responsible for recording the loss."""
    c = _Flaky(fail_times=99)
    assert call_json_with_retries(c, "m", [], retries=3) is None
    assert c.attempts == 3


# ── Endpoint inheritance ─────────────────────────────────────────────────────

def test_endpoints_inherit_down_the_chain():
    c = Config(book_id="1", base_url="http://gpu:8000/v1")
    assert c.validator_base_url == "http://gpu:8000/v1"
    assert c.structure_base_url == "http://gpu:8000/v1"


def test_structure_follows_the_validator_endpoint():
    """Structure is a judgment call, so it defaults to wherever the validator
    runs rather than to the bulk-work box."""
    c = Config(book_id="1", base_url="http://gpu:8000/v1",
               validator_base_url="http://mac:8000/v1")
    assert c.structure_base_url == "http://mac:8000/v1"
    assert c.validator_base_url == "http://mac:8000/v1"


def test_template_kwargs_are_per_endpoint(stub):
    """Chain-of-thought is waste on the structure pass — measured at 5,131
    completion tokens versus 316 for the same correct answer — but it may be
    worth having elsewhere, so it is set per endpoint rather than globally."""
    r = LLMRouter()
    r.register("http://gpu:8000/v1")
    r.register("http://mac:8000/v1", template_kwargs={"enable_thinking": False})
    assert r._client("small-model").template_kwargs == {}
    assert r._client("big-model").template_kwargs == {"enable_thinking": False}

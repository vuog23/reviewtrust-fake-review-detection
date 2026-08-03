import pytest

from src.groq_client import GroqClient, GroqUnavailableError, strip_reasoning
from tests.conftest import FakeGroqSDK


def test_closed_thinking_block_is_removed():
    assert strip_reasoning("<think>weighing it up</think>\n\n# Review assessment") == "# Review assessment"


def test_unclosed_thinking_block_is_removed():
    # A reply cut off by max_completion_tokens can leave <think> hanging open.
    assert strip_reasoning("# Review assessment\n<think>still weighing") == "# Review assessment"


def test_reply_without_thinking_is_untouched():
    assert strip_reasoning("  # Review assessment  ") == "# Review assessment"


def test_thinking_is_stripped_from_a_live_reply():
    sdk = FakeGroqSDK(["<think>internal</think>\n# Review assessment\n\nDecision: suspicious"])
    client = GroqClient(model="qwen/qwen3.6-27b", client=sdk)

    reply = client.complete([{"role": "user", "content": "hi"}], temperature=0.7)

    assert reply.startswith("# Review assessment")
    assert "internal" not in reply


def test_request_carries_the_generation_and_reasoning_settings():
    sdk = FakeGroqSDK(["ok"])
    client = GroqClient(model="qwen/qwen3.6-27b", client=sdk)

    client.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.4,
        top_p=0.8,
        max_completion_tokens=512,
    )

    request = sdk.requests[0]
    assert request["model"] == "qwen/qwen3.6-27b"
    assert request["temperature"] == 0.4
    assert request["top_p"] == 0.8
    assert request["max_completion_tokens"] == 512
    # Both are needed: qwen3.6 otherwise writes its chain of thought into content.
    assert request["reasoning_effort"] == "none"
    assert request["reasoning_format"] == "hidden"


def test_missing_api_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("REVIEWTRUST_TEST_KEY", raising=False)
    client = GroqClient(model="qwen/qwen3.6-27b", api_key_env="REVIEWTRUST_TEST_KEY")

    assert client.is_configured is False
    with pytest.raises(GroqUnavailableError, match="REVIEWTRUST_TEST_KEY"):
        client.complete([{"role": "user", "content": "hi"}])


def test_a_key_in_the_environment_counts_as_configured(monkeypatch):
    monkeypatch.setenv("REVIEWTRUST_TEST_KEY", "gsk_example")
    assert GroqClient(model="m", api_key_env="REVIEWTRUST_TEST_KEY").is_configured is True


@pytest.mark.parametrize(
    ("exception_name", "expected"),
    [
        ("AuthenticationError", "rejected the API key"),
        ("PermissionDeniedError", "not allowed to use"),
        ("NotFoundError", "does not serve a model"),
        ("RateLimitError", "rate limit"),
        ("APIConnectionError", "could not be reached"),
        ("APITimeoutError", "could not be reached"),
    ],
)
def test_sdk_failures_become_one_actionable_error(exception_name, expected):
    raised = type(exception_name, (Exception,), {})("boom")
    client = GroqClient(model="qwen/qwen3.6-27b", client=FakeGroqSDK([raised]))

    with pytest.raises(GroqUnavailableError, match=expected):
        client.complete([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("status", [500, 502, 503])
def test_provider_capacity_errors_tell_the_user_to_retry(status):
    """Groq answers 503 'currently over capacity' for preview models under load."""
    raised = type("InternalServerError", (Exception,), {"status_code": status})("over capacity")
    client = GroqClient(model="qwen/qwen3.6-27b", client=FakeGroqSDK([raised]))

    with pytest.raises(GroqUnavailableError, match="try again in a moment"):
        client.complete([{"role": "user", "content": "hi"}])


def test_provider_error_body_is_logged_but_not_handed_to_the_caller(caplog):
    """The route returns str(exc) to the browser, so provider detail stays in the log."""
    raised = type("InternalServerError", (Exception,), {"status_code": 503})("internal trace xyz123")
    client = GroqClient(model="qwen/qwen3.6-27b", client=FakeGroqSDK([raised]))

    with pytest.raises(GroqUnavailableError) as caught:
        client.complete([{"role": "user", "content": "hi"}])

    assert "internal trace xyz123" in caplog.text
    assert "internal trace xyz123" not in str(caught.value)


def test_truncated_reply_still_returns_what_arrived(caplog):
    sdk = FakeGroqSDK([("# Review assessment", "length")])
    client = GroqClient(model="qwen/qwen3.6-27b", client=sdk)

    assert client.complete([{"role": "user", "content": "hi"}]) == "# Review assessment"
    assert "cut short" in caplog.text

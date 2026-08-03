from pathlib import Path

from src.groq_client import GroqClient
from src.reasoner_inference import ReasonerInference
from src.utils import SYSTEM_PROMPT
from tests.conftest import FakeGroqSDK


CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config.yaml")


def _reasoner(replies=None):
    sdk = FakeGroqSDK(replies or ["# Review assessment"])
    client = GroqClient(model="qwen/qwen3.6-27b", client=sdk)
    return ReasonerInference(config_path=CONFIG_PATH, client=client), sdk


def test_project_config_selects_the_groq_model():
    reasoner, _ = _reasoner()
    assert reasoner.model_name == "qwen/qwen3.6-27b"


def test_system_prompt_and_payload_are_sent_as_two_messages():
    reasoner, sdk = _reasoner()

    reasoner.inference('{"task": "explain"}')

    messages = sdk.requests[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == '{"task": "explain"}'


def test_per_call_temperature_reaches_the_api():
    """This is the path production uses: PipelineService passes the UI slider value
    per call rather than assigning to the shared attribute, which would race now
    that the reasoner runs outside the pipeline lock."""
    reasoner, sdk = _reasoner()

    reasoner.inference("payload", temperature=0.21)

    assert sdk.requests[0]["temperature"] == 0.21
    assert reasoner.temperature != 0.21, "the shared default must not be mutated"


def test_configured_temperature_is_used_when_none_is_passed():
    reasoner, sdk = _reasoner()

    reasoner.temperature = 0.35
    reasoner.inference("payload")

    assert sdk.requests[0]["temperature"] == 0.35


def test_thinking_output_never_reaches_the_caller():
    reasoner, _ = _reasoner(["<think>chain of thought</think>\n# Review assessment"])

    assert reasoner.inference("payload") == "# Review assessment"


def test_oversized_payload_is_trimmed_before_sending():
    reasoner, sdk = _reasoner()
    reasoner.max_input_chars = 100

    reasoner.inference("x" * 5_000)

    assert len(sdk.requests[0]["messages"][1]["content"]) == 100


def test_missing_key_does_not_prevent_construction(monkeypatch):
    """The classifier and NER stages must still work with no Groq key present."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    reasoner = ReasonerInference(config_path=CONFIG_PATH)

    assert reasoner.is_configured is False

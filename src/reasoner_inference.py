import logging

import yaml
from src.groq_client import GroqClient
from src.utils import SYSTEM_PROMPT


LOGGER = logging.getLogger(__name__)


class ReasonerInference:
    """Explain a classifier decision with a hosted Qwen model on Groq Cloud."""

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self, config_path: str = "config.yaml", client: GroqClient | None = None):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        inference_config = config["reasoner_inference"]
        generation_config = inference_config["generation"]

        self.model_name = inference_config["model_name"]
        self.max_input_chars = int(inference_config.get("max_input_chars", 48_000))

        self.max_completion_tokens = int(generation_config["max_completion_tokens"])
        self.temperature = generation_config["temperature"]
        self.top_p = generation_config["top_p"]

        self._client = client or GroqClient(
            model=self.model_name,
            api_key_env=inference_config.get("api_key_env", "GROQ_API_KEY"),
            base_url=inference_config.get("base_url"),
            timeout=float(inference_config.get("timeout_seconds", 120)),
            max_retries=int(inference_config.get("max_retries", 3)),
            reasoning_effort=inference_config.get("reasoning_effort", "none"),
            reasoning_format=inference_config.get("reasoning_format", "hidden"),
        )
        if not self._client.is_configured:
            LOGGER.warning(
                "%s is not set. Classification and entity extraction will still run, but the "
                "explanation stage will fail until a Groq API key is available.",
                self._client.api_key_env,
            )

    @property
    def is_configured(self) -> bool:
        return self._client.is_configured

    def inference(self, prompt: str, temperature: float | None = None) -> str:
        if len(prompt) > self.max_input_chars:
            LOGGER.warning(
                "Trimming a %d character reasoner prompt to %d",
                len(prompt),
                self.max_input_chars,
            )
            prompt = prompt[: self.max_input_chars]

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return self._client.complete(
            messages,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )

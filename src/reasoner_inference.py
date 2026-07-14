import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import SYSTEM_PROMPT


class ReasonerInference:
    SYSTEM_PROMPT=SYSTEM_PROMPT
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        inference_config = config["reasoner_inference"]
        generation_config = inference_config["generation"]

        self.model_name = inference_config["model_name"]
        self.cache_dir = inference_config["cache_dir"]

        self.max_new_tokens = generation_config["max_new_tokens"]
        self.do_sample = generation_config["do_sample"]
        self.temperature = generation_config["temperature"]
        self.top_p = generation_config["top_p"]
        self.repetition_penalty = generation_config["repetition_penalty"]

        dtype_name = inference_config["torch_dtype"]

        if dtype_name == "float16":
            self.dtype = torch.float16
        elif dtype_name == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype_name == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = (
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=inference_config["trust_remote_code"],
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            device_map=inference_config["device_map"],
            low_cpu_mem_usage=inference_config["low_cpu_mem_usage"],
            trust_remote_code=inference_config["trust_remote_code"],
        )
        self.model.eval()

    def inference(self, prompt: str) -> str:
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

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        generation_arguments = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample:
            generation_arguments["temperature"] = self.temperature
            generation_arguments["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_arguments,
            )

        generated_ids = output_ids[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        response = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return response.strip()
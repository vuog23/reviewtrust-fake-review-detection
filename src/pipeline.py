import json

from src.classifier_inference import Classifierinference
from src.ner_inference import NERInference
from src.reasoner_inference import ReasonerInference


class InferencePipeline:
    def __init__(
        self,
        config_path: str = "config.yaml",
        classifier_model_name: str = None,
        classifier_loss_key: str = None,
    ):
        self.config_path = config_path
        self.classifier_model_name = classifier_model_name
        self.classifier_loss_key = classifier_loss_key

        self.ner = NERInference(config_path=config_path)

        self.ner.LABELS = self.ner.NER_LABELS

        self.reasoner = ReasonerInference(config_path=config_path)

    def inference(self, text: str) -> dict:
        classifier_output = Classifierinference(
            text=text,
            model_name=self.classifier_model_name,
            loss_key=self.classifier_loss_key,
            config_path=self.config_path,
        )

        ner_output = self.ner.predict(text)

        reasoner_prompt = json.dumps(
            {
                "task": (
                    "Analyze the original text using the classifier and NER "
                    "results, then return the final risk assessment."
                ),
                "original_text": text,
                "classifier_output": classifier_output,
                "ner_output": ner_output,
            },
            indent=2,
            ensure_ascii=False,
        )

        reasoner_response = self.reasoner.inference(reasoner_prompt)

        try:
            reasoner_output = json.loads(reasoner_response)
        except json.JSONDecodeError:
            reasoner_output = {
                "raw_response": reasoner_response,
            }

        return {
            "input_text": text,
            "classifier": classifier_output,
            "ner": ner_output,
            "reasoner": reasoner_output,
        }


# pipeline = InferencePipeline(
#     config_path="config.yaml",
# )

# while True:
#     text = input("\nEnter text: ").strip()

#     if text.lower() in {"exit", "quit"}:
#         break

#     if not text:
#         continue

#     output = pipeline.inference(text)

#     print(
#         json.dumps(
#             output,
#             indent=2,
#             ensure_ascii=False,
#         )
#     )
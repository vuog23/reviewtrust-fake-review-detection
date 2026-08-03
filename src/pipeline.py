from src.ner_inference import NERInference
from src.reasoner_inference import ReasonerInference


class InferencePipeline:
    """Holds the long-lived inference components.

    The classifier is deliberately absent: it is a plain function invoked per
    request, so only the entity and reasoning stages need to persist here.
    """

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
        self.reasoner = ReasonerInference(config_path=config_path)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class BERT:
    def __init__(
            self,
            model_path,
            num_labels: int = 2,
            cache_dir: str = None,
            trust_remote_code: bool = False,
            dropout: float = None,
            drop_path_rate: float = None,
            id2label: dict = {0: "non-deceptive", 1: "deceptive"},
            label2id: dict = {"non-deceptive": 0, "deceptive": 1}
    ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        model_config = AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        model_type = str(getattr(model_config, "model_type", "")).lower()

        if dropout is not None:
            dropout = float(dropout)

        dropout_fields = {
            "bert": (
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "classifier_dropout",
            ),
            "roberta": (
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "classifier_dropout",
            ),
            "distilbert": (
                "dropout",
                "attention_dropout",
                "seq_classif_dropout",
            ),
            "modernbert": (
                "attention_dropout",
                "embedding_dropout",
                "mlp_dropout",
                "classifier_dropout",
            ),
        }

        fields = dropout_fields.get(model_type)

        modified_fields = []
        for field in fields:
            setattr(model_config, field, dropout)
            modified_fields.append(field)

        if drop_path_rate is not None:
            drop_path_rate = float(drop_path_rate)

        model_config.drop_path_rate = drop_path_rate
        model_config.num_labels = num_labels
        model_config.id2label = id2label
        model_config.label2id = label2id

        self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=model_config,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
        )
"""Span post-processing, exercised without loading the extractor."""
import yaml
from pathlib import Path

from src.ner_inference import (
    NERInference,
    drop_filler_spans,
    resolve_contained_spans,
)


CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8"))


def _filterer(threshold=None):
    """A NERInference with only the filtering state populated -- no model, no GPU."""
    ner = NERInference.__new__(NERInference)
    settings = CONFIG["ner_inference"]
    ner.default_threshold = settings["threshold"]
    ner.threshold = settings["threshold"] if threshold is None else threshold
    ner.min_span_chars = settings["min_span_chars"]
    ner.label_thresholds = settings["label_thresholds"]
    return ner


def span(text, label, confidence, start=0, end=None):
    return {
        "text": text, "label": label, "confidence": confidence,
        "start": start, "end": len(text) if end is None else end,
    }


def test_noisy_labels_need_more_confidence_than_clean_ones():
    """order_or_transaction fires on 'price'/'Shipping' in ordinary reviews; the
    solicitation labels never did, so one global cut cannot serve both."""
    entities = [
        span("Shipping", "order_or_transaction", 0.66),
        span("gift card", "payment_or_reward", 0.40),
    ]

    kept = _filterer().filter_entities(entities)

    assert [e["text"] for e in kept] == ["gift card"]


def test_a_confident_noisy_span_still_survives():
    kept = _filterer().filter_entities([span("Order #A1938472", "order_or_transaction", 0.89)])
    assert len(kept) == 1


def test_slider_shifts_every_label_threshold_together():
    entities = [span("gift card", "payment_or_reward", 0.40)]

    # payment_or_reward is tuned to 0.35, so 0.40 clears it at the neutral position.
    assert _filterer().filter_entities(entities)
    # Sliding 0.20 stricter pushes the cut to 0.55 and drops it.
    assert not _filterer(threshold=0.70).filter_entities(entities)
    # Sliding looser keeps it.
    assert _filterer(threshold=0.30).filter_entities(entities)


def test_pronouns_and_stubs_are_dropped():
    entities = [
        span("it", "suspicious_claim", 0.9),
        span("I", "person_or_account", 0.9),
        span("the", "review_instruction", 0.9),
        span("gift card", "payment_or_reward", 0.9),
    ]

    assert [e["text"] for e in drop_filler_spans(entities, 3)] == ["gift card"]


def test_same_span_under_two_labels_keeps_the_confident_one():
    """The extractor returned 'Michael' as both a person and a brand."""
    entities = [
        span("Michael", "person_or_account", 0.99, 7, 14),
        span("Michael", "product_or_brand", 0.80, 7, 14),
    ]

    kept = resolve_contained_spans(entities)

    assert len(kept) == 1
    assert kept[0]["label"] == "person_or_account"


def test_a_weak_span_inside_a_strong_one_is_dropped():
    entities = [
        span("remove my 1 star review", "review_instruction", 0.86, 46, 69),
        span("remove", "pressure_or_manipulation", 0.23, 46, 52),
    ]

    kept = resolve_contained_spans(entities)

    assert [e["text"] for e in kept] == ["remove my 1 star review"]


def test_a_sharper_span_inside_a_weaker_one_survives():
    """'1 star review' is the more precise evidence, so it is not swallowed."""
    entities = [
        span("remove my 1 star review", "review_instruction", 0.86, 46, 69),
        span("1 star review", "rating_instruction", 0.90, 56, 69),
    ]

    kept = resolve_contained_spans(entities)

    assert {e["text"] for e in kept} == {"remove my 1 star review", "1 star review"}


def test_neighbouring_spans_are_both_kept():
    entities = [
        span("deals@example.com", "contact_information", 0.9, 10, 27),
        span("+1-555-0134", "contact_information", 0.9, 40, 51),
    ]

    assert len(resolve_contained_spans(entities)) == 2


def test_product_or_brand_is_no_longer_extracted():
    """It scored 0.98+ on 'headphones' and 'blender' in genuine reviews, above its
    own scores on fraudulent ones, so it buried the real signal."""
    assert "product_or_brand" not in NERInference.NER_LABELS


def test_every_label_has_a_tuned_threshold():
    assert set(CONFIG["ner_inference"]["label_thresholds"]) == set(NERInference.NER_LABELS)

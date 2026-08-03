"""The presentation layer over the reasoner's Markdown."""
from app.services.markdown_service import MarkdownService, _is_source_evidence


REVIEW = (
    "I bought these headphones last week and the sound quality is decent for the price. "
    "Battery lasts about six hours. Shipping took three days. Would recommend to a friend."
)

CONTRACT_REPLY = """# Review assessment

**Decision:** legitimate
**Risk level:** none
**Confidence:** 91%

## Summary

The calibrated classifier assessed this review as non-deceptive with high confidence. The text
describes concrete, checkable product details and contains no request or incentive of any kind.

## Evidence

* **"the sound quality is decent for the price"** — A measured judgement rather than absolute praise.
* **"Battery lasts about six hours"** — A concrete, falsifiable detail about real use.

## Recommended action

No action required; publish normally.
"""


def _classifier(label="Non-Deceptive", confidence=0.91, accepted=True):
    return {
        "calibrated": {"label": label, "confidence": confidence},
        "uncalibrated": {"label": label, "confidence": confidence},
        "selective_prediction": {
            "confidence_threshold": 0.90, "accepted": accepted,
            "decision": "accepted" if accepted else "rejected",
        },
    }


def _present(reply=CONTRACT_REPLY, classifier=None, review=REVIEW):
    return MarkdownService().presentation_markdown(
        reply, original_text=review, classifier_output=classifier or _classifier()
    )


def test_summary_survives_even_though_it_names_the_classifier():
    """The summary explains the classifier's decision, so it necessarily mentions it.
    A filter meant to hide raw dumps used to delete the whole paragraph."""
    markdown = _present()

    assert "The calibrated classifier assessed this review as non-deceptive" in markdown
    assert "### Summary" in markdown


def test_quoted_evidence_survives_the_grounding_check():
    """The system prompt asks for a quote, so the model emits quotation marks."""
    markdown = _present()

    assert "the sound quality is decent for the price" in markdown
    assert "Battery lasts about six hours" in markdown
    # The wrapping quotes are removed rather than the whole bullet.
    assert '**"the sound quality' not in markdown


def test_every_section_of_the_contract_reaches_the_reader():
    markdown = _present()

    for heading in ("### Summary", "### Evidence", "### Recommended action"):
        assert heading in markdown
    body = markdown.split("### Evidence", 1)[1]
    assert body.strip().startswith("-"), "the Evidence section must not be empty"


def test_the_classifier_owns_the_decision_not_the_model():
    """The reply says 'legitimate'; the classifier says the label and holds authority."""
    markdown = _present(classifier=_classifier(label="Deceptive", confidence=0.97))

    assert markdown.startswith("## Deceptive")
    assert "Classifier confidence: 97.0%" in markdown


def test_a_rejected_prediction_is_shown_as_uncertain():
    markdown = _present(classifier=_classifier(accepted=False))

    assert markdown.startswith("## Uncertain")
    assert "classifier leaned toward" in markdown.lower()
    assert "avoid automatic enforcement" in markdown.lower()


def test_hallucinated_quotes_are_still_withheld():
    reply = CONTRACT_REPLY.replace(
        '**"Battery lasts about six hours"** — A concrete, falsifiable detail about real use.',
        '**"send me a gift card for a 5 star review"** — The reviewer was offered a reward.',
    )

    markdown = _present(reply=reply)

    assert "gift card" not in markdown
    assert "the sound quality is decent for the price" in markdown


def test_an_evidence_section_emptied_by_grounding_says_so():
    reply = CONTRACT_REPLY.replace(
        '**"the sound quality is decent for the price"** — A measured judgement rather than absolute praise.',
        '**"contact me at fraud@example.com"** — An off-platform contact attempt.',
    ).replace(
        '**"Battery lasts about six hours"** — A concrete, falsifiable detail about real use.',
        '**"I was paid for this"** — An admission of compensation.',
    )

    markdown = _present(reply=reply)

    assert "does not appear in this review" in markdown
    assert "fraud@example.com" not in markdown


def test_an_unbolded_bullet_is_grounded_too():
    """The model does not always bold its quote; that must not bypass grounding."""
    reply = CONTRACT_REPLY.replace(
        '* **"the sound quality is decent for the price"** — A measured judgement rather than absolute praise.',
        '* "the sound quality is decent for the price" — A measured judgement.',
    ).replace(
        '* **"Battery lasts about six hours"** — A concrete, falsifiable detail about real use.',
        '* "they paid me to write this" — An admission of compensation.',
    )

    markdown = _present(reply=reply)

    assert "the sound quality is decent for the price" in markdown
    assert "paid me to write this" not in markdown


def test_a_close_paraphrase_across_a_line_break_is_accepted():
    assert _is_source_evidence("sound quality is decent   for the price", REVIEW)


def test_quotation_marks_no_longer_defeat_grounding():
    assert _is_source_evidence('"Battery lasts about six hours"', REVIEW)
    assert _is_source_evidence("`Shipping took three days`", REVIEW)
    assert not _is_source_evidence('"a gift card for a five star review"', REVIEW)


def test_a_malformed_reply_still_falls_back_to_salvage():
    """Responses that ignore the layout must keep going through the old path."""
    markdown = MarkdownService().presentation_markdown(
        '{"decision":"suspicious","risk_level":"high","summary":"A reward was offered."}',
        original_text=REVIEW,
        classifier_output=_classifier(label="Deceptive", accepted=True),
    )

    assert "A reward was offered." in markdown
    assert "{" not in markdown

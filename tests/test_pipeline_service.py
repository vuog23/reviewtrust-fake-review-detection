import threading


def test_request_thresholds_reach_each_existing_component(fake_service):
    result = fake_service.analyze("A gift card offer", 0.90, 0.63, 0.35)
    prompt = fake_service._pipeline.reasoner.last_prompt

    assert result["settings"] == {
        "classifier_threshold": 0.90, "ner_threshold": 0.63, "temperature": 0.35,
    }
    assert result["ner"]["threshold"] == 0.63
    assert fake_service._pipeline.reasoner.used_temperature == 0.35
    assert "classifier is the prediction source" in prompt["task"]
    assert "selective_prediction.accepted" in prompt["task"]
    assert prompt["original_text"] == "A gift card offer"
    assert prompt["classifier_output"]["selective_prediction"]["accepted"] is False
    # NER still runs and still drives the UI highlighting, but the reasoner is not
    # shown it: it reasons from the review and the calibrated verdict alone.
    assert "ner_output" not in prompt
    assert result["ner"]["entities"]


def test_reasoner_runs_outside_the_pipeline_lock(fake_service):
    """The lock guards the two local GPU models; the hosted call must not serialize.

    The barrier can only be satisfied if both threads are inside inference() at the
    same moment. If the reasoner were still inside the lock they could never be, and
    the barrier would time out.
    """
    barrier = threading.Barrier(2, timeout=5)

    def concurrent_reasoner(prompt, temperature=None):
        barrier.wait()
        return "## Evidence summary\n\nBoth requests reached the reasoner together."

    fake_service._pipeline.reasoner.inference = concurrent_reasoner
    failures = []

    def run():
        try:
            fake_service.analyze("A gift card offer", 0.90, 0.50, 0.70)
        except Exception as exc:  # BrokenBarrierError if they were serialized
            failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures, f"requests were serialized behind the lock: {failures}"


def test_reasoner_prompt_sends_the_review_once(fake_service):
    review = "A gift card offer. " * 40
    fake_service.analyze(review, 0.90, 0.50, 0.70)
    prompt = fake_service._pipeline.reasoner.last_prompt

    assert prompt["original_text"] == review
    # The classifier output echoes the review back; the reasoner must not be billed
    # for the same text twice.
    assert "text" not in prompt["classifier_output"]


def test_reasoner_prompt_is_bounded_for_long_reviews(fake_service):
    from app.services.pipeline_service import REASONER_TEXT_BUDGET

    review = "A gift card offer. " * 2_000
    result = fake_service.analyze(review, 0.90, 0.50, 0.70)
    prompt = fake_service._pipeline.reasoner.last_prompt

    assert len(review) > REASONER_TEXT_BUDGET
    assert len(prompt["original_text"]) < len(review)
    assert prompt["original_text"].endswith("[…truncated]")
    # Truncation is for the model's benefit only; the caller still sees its review.
    assert result["input_text"] == review


def test_reasoner_prompt_hides_calibration_metadata(fake_service):
    fake_service.analyze("A gift card offer", 0.90, 0.50, 0.70)
    prompt = fake_service._pipeline.reasoner.last_prompt

    assert set(prompt) == {"task", "original_text", "classifier_output"}
    assert set(prompt["classifier_output"]) == {
        "uncalibrated", "calibrated", "selective_prediction",
    }


def test_system_prompt_no_longer_instructs_the_model_about_ner():
    from src.utils import SYSTEM_PROMPT

    assert "NER" not in SYSTEM_PROMPT
    assert "Interpret NER output" not in SYSTEM_PROMPT
    # The risk taxonomy stays: the model still needs the vocabulary to name findings.
    assert "review_compensation" in SYSTEM_PROMPT
    assert "pressure_or_manipulation" in SYSTEM_PROMPT


def test_markdown_html_is_sanitized(fake_service):
    result = fake_service.analyze("A gift card offer", 0.90, 0.50, 0.70)
    html = result["reasoner"]["html"]
    assert "<h2>Uncertain assessment</h2>" in html
    assert "Potential signals for manual review" in html
    assert "payment, incentive, gift, or reward" in html
    assert "<script" not in html


def test_json_reasoner_response_hides_classifier_details(fake_service):
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        '{"decision":"likely_suspicious","risk_level":"high",'
        '"summary":"The offer attempts to influence a review.",'
        '"supporting_evidence":[{"text":"gift card","explanation":"A reward is linked to feedback."}],'
        '"recommended_action":"Send for human review.","uncertainties":[],'
        '"classifier_assessment":{"predicted_label":"FAKE","calibrated_confidence":0.99}}'
    )
    result = fake_service.analyze("A gift card offer", 0.80, 0.50, 0.70)
    html = result["reasoner"]["html"]
    assert "<h2>Fake</h2>" in html
    assert "gift card" in html
    assert "classifier" not in html.lower()
    assert "0.99" not in html


def test_json_after_reasoner_intro_is_formatted_not_escaped(fake_service):
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        'Based on the provided output, here is the final risk assessment:\n\n'
        '{"decision":"suspicious","risk_level":"high",'
        '"summary":"The reward is linked to a requested review.",'
        '"supporting_evidence":[],"recommended_action":"Review manually.",'
        '"classifier_assessment":{"calibrated_confidence":0.91}}'
    )
    html = fake_service.analyze("A gift card offer", 0.80, 0.50, 0.70)["reasoner"]["html"]
    assert "<h2>Fake</h2>" in html
    assert "The reward is linked" in html
    assert "&amp;quot;" not in html
    assert "classifier_assessment" not in html


def test_alternative_reasoner_dictionary_becomes_description(fake_service):
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        '{"final_risk_assessment":"Highly suspicious",'
        '"reasoning":"Several exaggerated claims are presented as guaranteed facts.",'
        '"risk_categories":["deceptive_promotion","suspicious_claim"]}'
    )
    html = fake_service.analyze("A gift card offer", 0.80, 0.50, 0.70)["reasoner"]["html"]
    assert "<h2>Fake</h2>" in html
    assert "Several exaggerated claims" in html
    assert "final_risk_assessment" not in html
    assert "{" not in html


def test_fallback_removes_dictionary_before_rendering(fake_service):
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        'Final result: {"unexpected": "value", broken}\n\n'
        '### Why\nThe review makes an unsupported guarantee.'
    )
    html = fake_service.analyze("A gift card offer", 0.90, 0.50, 0.70)["reasoner"]["html"]
    assert "Uncertain assessment" in html
    assert "unsupported guarantee" not in html
    assert "unexpected" not in html
    assert "{" not in html


def test_html_encoded_truncated_json_becomes_user_facing_assessment(fake_service):
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        '{\n  &quot;decision&quot;: &quot;high&quot;,\n'
        '  &quot;risk_level&quot;: &quot;critical&quot;,\n'
        '  &quot;confidence&quot;: 0.725,\n'
        '  &quot;summary&quot;: &quot;The review makes several unsupported claims.&quot;,\n'
        '  &quot;detected_behaviors&quot;: [&quot;payment_offer&quot;, &quot;suspicious_claim&quot;],\n'
        '  &quot;supporting_evidence&quot;: [{&quot;text&quot;: &quot;unfinished'
    )
    html = fake_service.analyze("A gift card offer", 0.80, 0.50, 0.70)["reasoner"]["html"]
    assert "<h2>Fake</h2>" in html
    assert "Confidence: 72.5%" in html
    assert "unsupported claims" in html
    assert "&amp;quot;" not in html
    assert "{" not in html


def test_plain_truncated_json_never_reaches_stage_three(fake_service):
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        '{"final_risk_assessment":"suspicious","risk_category":"product_or_brand",'
        '"risk_level":"high","confidence":0.786,'
        '"summary":"The review contains several conflicting product claims.",'
        '"detected_behaviors":["deceptive","misleading_claims"],'
        '"supporting_evidence":[{"text":"completely unusable",'
        '"explanation":"The statement conflicts with the later positive assessment'
    )
    html = fake_service.analyze("Conflicting review", 0.80, 0.50, 0.70)["reasoner"]["html"]
    assert "<h2>Fake</h2>" in html
    assert "Confidence: 78.6%" in html
    assert "conflicting product claims" in html
    assert "final_risk_assessment" not in html
    assert "{" not in html


def test_canonical_system_prompt_requires_markdown_not_json():
    from src.utils import SYSTEM_PROMPT

    assert "Return only concise Markdown intended for an end user." in SYSTEM_PROMPT
    assert "Never return JSON" in SYSTEM_PROMPT
    assert "# Review assessment" in SYSTEM_PROMPT
    assert "Return only one valid JSON object" not in SYSTEM_PROMPT


def test_rejected_classifier_overrides_hallucinated_reasoner_risk(fake_service):
    review = "I highly recommend this watch to everyone."
    fake_service._pipeline.reasoner.inference = lambda *_, **__: (
        '{"decision":"suspicious","risk_level":"high",'
        '"summary":"The message requests a rating.",'
        '"detected_behaviors":["request_for_rating"],'
        '"supporting_evidence":[{"text":"Request for a rating:",'
        '"explanation":"The reviewer asks for a rating."}]}'
    )
    html = fake_service.analyze(review, 0.90, 0.50, 0.70)["reasoner"]["html"]
    assert "<h2>Uncertain</h2>" in html
    assert "request for a rating" not in html.lower()
    assert "classifier leaned toward" in html.lower()
    assert "avoid automatic enforcement" in html.lower()

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
    assert prompt["ner_output"]["entities"]


def test_markdown_html_is_sanitized(fake_service):
    result = fake_service.analyze("A gift card offer", 0.90, 0.50, 0.70)
    html = result["reasoner"]["html"]
    assert "<h2>Uncertain assessment</h2>" in html
    assert "Potential signals for manual review" in html
    assert "payment, incentive, gift, or reward" in html
    assert "<script" not in html


def test_json_reasoner_response_hides_classifier_details(fake_service):
    fake_service._pipeline.reasoner.inference = lambda _: (
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
    fake_service._pipeline.reasoner.inference = lambda _: (
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
    fake_service._pipeline.reasoner.inference = lambda _: (
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
    fake_service._pipeline.reasoner.inference = lambda _: (
        'Final result: {"unexpected": "value", broken}\n\n'
        '### Why\nThe review makes an unsupported guarantee.'
    )
    html = fake_service.analyze("A gift card offer", 0.90, 0.50, 0.70)["reasoner"]["html"]
    assert "Uncertain assessment" in html
    assert "unsupported guarantee" not in html
    assert "unexpected" not in html
    assert "{" not in html


def test_html_encoded_truncated_json_becomes_user_facing_assessment(fake_service):
    fake_service._pipeline.reasoner.inference = lambda _: (
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
    fake_service._pipeline.reasoner.inference = lambda _: (
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


def test_rejected_classifier_overrides_hallucinated_llama_risk(fake_service):
    review = "I highly recommend this watch to everyone."
    fake_service._pipeline.reasoner.inference = lambda _: (
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

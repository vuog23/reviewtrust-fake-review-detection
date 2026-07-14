from __future__ import annotations

import json
import re
from html import unescape
from typing import Any

import bleach
from markdown_it import MarkdownIt


ALLOWED_TAGS = {
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3",
    "h4", "h5", "h6", "hr", "li", "ol", "p", "pre", "strong", "table",
    "tbody", "td", "th", "thead", "tr", "ul",
}
ALLOWED_ATTRIBUTES = {"a": ["href", "title", "rel", "target"]}


def _secure_link(attributes: dict[tuple[str | None, str], str], _: bool) -> dict:
    href = attributes.get((None, "href"), "")
    if href.startswith(("http://", "https://")):
        attributes[(None, "target")] = "_blank"
        attributes[(None, "rel")] = "noopener noreferrer nofollow"
    return attributes


class MarkdownService:
    def __init__(self) -> None:
        self._renderer = MarkdownIt(
            "commonmark",
            {"html": False, "linkify": False, "typographer": False},
        ).enable("table")

    def render(self, markdown_text: str) -> str:
        rendered = self._renderer.render(markdown_text or "")
        cleaned = bleach.clean(
            rendered,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            protocols={"http", "https", "mailto"},
            strip=True,
        )
        return bleach.linkify(
            cleaned,
            callbacks=[_secure_link],
            skip_tags={"pre", "code"},
            parse_email=False,
        )

    def presentation_markdown(
        self,
        raw_response: str,
        original_text: str = "",
        classifier_output: dict[str, Any] | None = None,
        ner_output: dict[str, Any] | None = None,
    ) -> str:
        """Create a concise user-facing explanation without exposing model dumps."""
        raw_response = unescape(raw_response or "")
        payload = _extract_json_object(raw_response)
        if payload is None:
            payload = _extract_partial_payload(raw_response)
        if payload is None:
            cleaned_markdown = _remove_internal_sections(raw_response)
            return _format_markdown_with_authority(
                cleaned_markdown,
                original_text,
                classifier_output,
                ner_output,
            )

        if not isinstance(payload, dict):
            return _remove_internal_sections(raw_response)

        authoritative_decision = _authoritative_decision(classifier_output)
        decision = authoritative_decision or str(
            payload.get("decision")
            or payload.get("final_risk_assessment")
            or payload.get("assessment")
            or payload.get("verdict")
            or "Assessment complete"
        )
        risk_level = "" if decision == "uncertain" else str(payload.get("risk_level", "")).strip()
        normalized_decision = decision.replace("_", " ").strip()
        if risk_level and normalized_decision.lower() in {
            "none", "low", "medium", "high", "critical"
        }:
            title = f"{risk_level.replace('_', ' ').strip().title()} risk"
        else:
            title = normalized_decision.title()
        lines = [f"## {title}"]
        metadata: list[str] = []
        if risk_level and "risk" not in title.lower():
            metadata.append(f"Risk level: {risk_level.replace('_', ' ').title()}")
        confidence = payload.get("confidence")
        if decision != "uncertain" and isinstance(confidence, (int, float)):
            metadata.append(f"Confidence: {float(confidence) * 100:.1f}%")
        if metadata:
            lines.append(f"**{' · '.join(metadata)}**")

        if decision == "uncertain" and classifier_output:
            label = str(classifier_output.get("calibrated", {}).get("label", "prediction"))
            summary = (
                f"The classifier leaned toward **{label.replace('_', ' ')}**, but its "
                "confidence did not meet the selected acceptance threshold. This review "
                "should be treated as uncertain and checked by a person rather than "
                "automatically enforced."
            )
        else:
            summary = str(
                payload.get("summary")
                or payload.get("reasoning")
                or payload.get("explanation")
                or payload.get("description")
                or ""
            ).strip()
        if summary:
            lines.extend(["", summary])

        indicators = (
            payload.get("detected_behaviors")
            or payload.get("risk_categories")
            or payload.get("selected_risk_indicators")
            or []
        )
        allowed_labels = {
            str(label).lower().replace("_", " ")
            for label in (ner_output or {}).get("risk_labels", [])
        }
        grounded_indicators = [
            item for item in indicators
            if ner_output is None
            or str(item).lower().replace("_", " ") in allowed_labels
        ] if isinstance(indicators, list) else []
        if grounded_indicators:
            lines.extend(["", "### Signals detected"])
            lines.extend(
                f"- {str(item).replace('_', ' ').capitalize()}"
                for item in grounded_indicators[:5]
                if str(item).strip()
            )

        evidence = payload.get("supporting_evidence", [])
        grounded_evidence = [
            item for item in evidence
            if isinstance(item, dict)
            and _is_source_evidence(str(item.get("text", "")), original_text)
        ] if isinstance(evidence, list) else []
        if grounded_evidence:
            lines.extend(["", "### Supporting evidence"])
            for item in grounded_evidence[:4]:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                explanation = str(item.get("explanation", "")).strip()
                if text and explanation:
                    lines.append(f"- **{text}** - {explanation}")
                elif text or explanation:
                    lines.append(f"- {text or explanation}")

        action = (
            "Review the text manually and avoid automatic enforcement."
            if decision == "uncertain"
            else str(payload.get("recommended_action", "")).strip()
        )
        if action:
            lines.extend(["", "### Recommended action", action])

        uncertainties = [] if decision == "uncertain" else payload.get("uncertainties", [])
        if isinstance(uncertainties, list) and uncertainties:
            lines.extend(["", "### Keep in mind"])
            lines.extend(f"- {item}" for item in uncertainties[:3] if str(item).strip())

        return "\n".join(lines).strip()


def _extract_json_object(raw_response: str) -> dict[str, Any] | None:
    """Parse a JSON object even when a small model adds text before or after it."""
    if not isinstance(raw_response, str):
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw_response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _authoritative_decision(classifier_output: dict[str, Any] | None) -> str | None:
    if not classifier_output:
        return None
    selective = classifier_output.get("selective_prediction", {})
    if selective.get("accepted") is False:
        return "uncertain"
    label = str(classifier_output.get("calibrated", {}).get("label", "")).strip()
    return label or None


def _format_markdown_with_authority(
    markdown_text: str,
    original_text: str,
    classifier_output: dict[str, Any] | None,
    ner_output: dict[str, Any] | None,
) -> str:
    decision = _authoritative_decision(classifier_output)
    if not decision:
        return markdown_text

    grounded_bullets: list[str] = []
    for line in markdown_text.splitlines():
        match = re.match(r"^\s*[-*]\s+\*\*(.+?)\*\*\s*[-–—:]\s*(.+)$", line)
        if match and _is_source_evidence(match.group(1), original_text):
            grounded_bullets.append(f"- **{match.group(1)}** - {match.group(2)}")

    if decision == "uncertain":
        label = str((classifier_output or {}).get("calibrated", {}).get("label", "prediction"))
        potential_signals = _grounded_ner_bullets(ner_output, original_text)
        lines = [
            "## Uncertain assessment",
            "",
            f"The classifier leaned toward **{label.replace('_', ' ')}**, but did not "
            "meet the selected acceptance threshold. The result is not confident enough "
            "for an automatic decision.",
            "",
            "### Potential signals for manual review",
        ]
        if grounded_bullets:
            lines.extend(grounded_bullets[:4])
        elif potential_signals:
            lines.extend(potential_signals[:4])
        else:
            lines.append("No source-grounded suspicious evidence was provided by the reasoner.")
        lines.extend([
            "",
            "### Recommended action",
            "Review the text manually and avoid automatic enforcement.",
        ])
        return "\n".join(lines)

    filtered: list[str] = [f"## {decision.replace('_', ' ').title()}", ""]
    in_evidence = False
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+review assessment", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\*\*(decision|risk level|confidence):", stripped, re.IGNORECASE):
            continue
        if re.match(r"^#{1,6}\s+evidence", stripped, re.IGNORECASE):
            in_evidence = True
            filtered.extend(["", "### Evidence"])
            continue
        if in_evidence and stripped.startswith(("- ", "* ")):
            match = re.match(r"^[-*]\s+\*\*(.+?)\*\*\s*[-–—:]\s*(.+)$", stripped)
            if match and _is_source_evidence(match.group(1), original_text):
                filtered.append(f"- **{match.group(1)}** - {match.group(2)}")
            continue
        if stripped.startswith("## ") and "evidence" not in stripped.lower():
            in_evidence = False
        filtered.append(line)
    return "\n".join(filtered).strip()


_NER_SIGNAL_EXPLANATIONS = {
    "review_compensation": "This may indicate compensation connected to review activity.",
    "rating_instruction": "This may request or specify a particular product rating.",
    "review_instruction": "This may instruct someone to submit, change, or remove a review.",
    "pressure_or_manipulation": "This wording may create urgency, pressure, or manipulation.",
    "suspicious_claim": "This may be an exaggerated, absolute, or unverifiable product claim.",
    "payment_or_reward": "This may refer to a payment, incentive, gift, or reward.",
    "external_link": "This may direct the reviewer to an external website.",
    "contact_information": "This may move communication to a private contact channel.",
    "order_or_transaction": "This may expose or request transaction-related information.",
}
_NER_SIGNAL_PRIORITY = tuple(_NER_SIGNAL_EXPLANATIONS)


def _grounded_ner_bullets(
    ner_output: dict[str, Any] | None,
    original_text: str,
) -> list[str]:
    """Turn exact NER spans into cautious evidence when Llama output is unusable."""
    if not ner_output:
        return []
    priority = {label: index for index, label in enumerate(_NER_SIGNAL_PRIORITY)}
    selected: dict[str, dict[str, Any]] = {}
    for entity in ner_output.get("entities", []):
        if not isinstance(entity, dict):
            continue
        text = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "")).strip()
        if label not in _NER_SIGNAL_EXPLANATIONS or not _is_source_evidence(text, original_text):
            continue
        key = " ".join(text.casefold().split())
        current = selected.get(key)
        if current is None or priority[label] < priority[str(current.get("label", ""))]:
            selected[key] = entity

    ordered = sorted(
        selected.values(),
        key=lambda item: (
            priority.get(str(item.get("label", "")), len(priority)),
            -float(item.get("confidence", 0.0)),
        ),
    )
    return [
        f"- **{str(entity.get('text', '')).strip()}** - "
        f"{_NER_SIGNAL_EXPLANATIONS[str(entity.get('label', ''))]}"
        for entity in ordered
    ]


def _is_source_evidence(evidence_text: str, original_text: str) -> bool:
    def normalize(value: str) -> str:
        value = value.casefold().replace("’", "'").replace("“", '"').replace("”", '"')
        return " ".join(value.strip().strip(":-–— ").split())

    evidence = normalize(evidence_text)
    source = normalize(original_text)
    return len(evidence) >= 3 and evidence in source


def _extract_partial_payload(raw_response: str) -> dict[str, Any] | None:
    """Recover useful top-level fields from JSON truncated during generation."""
    payload: dict[str, Any] = {}
    string_fields = (
        "decision", "final_risk_assessment", "assessment", "verdict",
        "risk_level", "summary", "reasoning", "explanation", "description",
        "recommended_action",
    )
    for field in string_fields:
        match = re.search(
            rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            raw_response,
            re.DOTALL,
        )
        if match:
            try:
                payload[field] = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                payload[field] = match.group(1)

    confidence = re.search(
        r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
        raw_response,
    )
    if confidence:
        payload["confidence"] = float(confidence.group(1))

    for field in ("detected_behaviors", "risk_categories", "selected_risk_indicators"):
        match = re.search(
            rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]',
            raw_response,
            re.DOTALL,
        )
        if not match:
            continue
        items = re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1))
        payload[field] = [
            json.loads(f'"{item}"') if "\\" in item else item
            for item in items
        ]

    if not any(
        payload.get(field)
        for field in ("decision", "final_risk_assessment", "summary", "reasoning")
    ):
        return None
    return payload


def _remove_internal_sections(markdown_text: str) -> str:
    """Hide verbose classifier/NER dumps from non-JSON fallback responses."""
    markdown_text = _strip_object_blocks(markdown_text)
    technical = re.compile(
        r"\b(classifier|ner|named entit(?:y|ies)|model output|calibrated|"
        r"uncalibrated|probabilit(?:y|ies)|label id)\b",
        re.IGNORECASE,
    )
    heading = re.compile(r"^\s*(?:#{1,6}\s+|\*\*[^*]+\*\*\s*$)")
    kept: list[str] = []
    hiding = False
    for line in (markdown_text or "").splitlines():
        if heading.match(line):
            hiding = bool(technical.search(line))
        if not hiding and not technical.search(line):
            kept.append(line)

    cleaned = "\n".join(kept).strip()
    if cleaned:
        return cleaned
    return "Analysis completed, but the explanation contained only internal model details."


def _strip_object_blocks(text: str) -> str:
    """Remove JSON-like object blocks so raw dictionaries never reach the UI."""
    output: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    for character in text or "":
        if depth == 0:
            if character == "{":
                depth = 1
                in_string = False
                escaped = False
            else:
                output.append(character)
            continue

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1

    return "".join(output).strip()

from __future__ import annotations

from typing import Any


def segment_entities(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split text at entity boundaries while retaining every overlapping entity."""
    valid_entities: list[dict[str, Any]] = []
    for entity in entities:
        start = int(entity.get("start", -1))
        end = int(entity.get("end", -1))
        if 0 <= start < end <= len(text):
            valid_entities.append(entity)

    boundaries = {0, len(text)}
    for entity in valid_entities:
        boundaries.update((int(entity["start"]), int(entity["end"])))

    points = sorted(boundaries)
    segments: list[dict[str, Any]] = []
    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        covering = [
            entity
            for entity in valid_entities
            if int(entity["start"]) <= start and int(entity["end"]) >= end
        ]
        covering.sort(
            key=lambda entity: (
                str(entity.get("label", "")),
                -float(entity.get("confidence", 0.0)),
                int(entity["start"]),
                int(entity["end"]),
            )
        )
        segments.append(
            {
                "text": text[start:end],
                "start": start,
                "end": end,
                "entities": covering,
            }
        )
    return segments


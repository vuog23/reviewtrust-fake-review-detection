from app.services.entity_segments import segment_entities


def entity(text, label, confidence, start, end):
    return {"text": text, "label": label, "confidence": confidence, "start": start, "end": end}


def test_repeated_text_uses_offsets_not_search():
    text = "free item then free item"
    segments = segment_entities(text, [entity("free", "reward", .9, 15, 19)])
    highlighted = [part for part in segments if part["entities"]]
    assert [(part["start"], part["text"]) for part in highlighted] == [(15, "free")]


def test_adjacent_spans_remain_separate():
    text = "giftcard"
    segments = segment_entities(text, [
        entity("gift", "reward", .9, 0, 4), entity("card", "payment", .8, 4, 8),
    ])
    assert [(part["text"], [e["label"] for e in part["entities"]]) for part in segments] == [
        ("gift", ["reward"]), ("card", ["payment"]),
    ]


def test_overlapping_spans_retain_all_evidence():
    text = "five star review"
    segments = segment_entities(text, [
        entity("five star", "rating_instruction", .9, 0, 9),
        entity("star review", "review_instruction", .8, 5, 16),
    ])
    overlap = next(part for part in segments if part["start"] == 5 and part["end"] == 9)
    assert {e["label"] for e in overlap["entities"]} == {"rating_instruction", "review_instruction"}


def test_unicode_offsets_are_preserved():
    text = "Nhận quà 🎁 miễn phí"
    start = text.index("🎁")
    segments = segment_entities(text, [entity("🎁 miễn phí", "reward", .95, start, len(text))])
    assert "".join(part["text"] for part in segments) == text
    assert next(part for part in segments if part["entities"])["start"] == start


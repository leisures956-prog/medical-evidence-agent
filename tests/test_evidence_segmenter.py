from app.evidence_segmenter import segment_abstract


def test_abstract_is_split_with_exact_offsets():
    text = "BACKGROUND: Sepsis is serious. RESULTS: AUC was 0.641. CONCLUSIONS: Use with care."

    segments = segment_abstract(text)

    assert [item.text for item in segments] == [
        "BACKGROUND: Sepsis is serious.",
        "RESULTS: AUC was 0.641.",
        "CONCLUSIONS: Use with care.",
    ]
    assert all(text[item.start:item.end] == item.text for item in segments)


def test_common_abbreviation_does_not_create_false_sentence():
    text = "Several markers, e.g. PCT and IL-18, were tested. Results were reported."

    segments = segment_abstract(text)

    assert len(segments) == 2
    assert segments[0].text.startswith("Several markers")


def test_empty_abstract_returns_no_segments():
    assert segment_abstract("  ") == []

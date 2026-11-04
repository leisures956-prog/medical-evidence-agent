from app.text_normalization import normalize_for_prompt


def test_html_entities_and_tags_are_cleaned_for_prompt_only():
    raw = "&lt;b&gt;Results:&lt;/b&gt;  PCT was elevated.\n\nIL-18 was elevated."

    cleaned = normalize_for_prompt(raw)

    assert cleaned == "Results: PCT was elevated. IL-18 was elevated."
    assert raw.startswith("&lt;b&gt;")


def test_plain_scientific_text_is_preserved():
    assert normalize_for_prompt("AUC = 0.641; p < 0.05.") == "AUC = 0.641; p < 0.05."

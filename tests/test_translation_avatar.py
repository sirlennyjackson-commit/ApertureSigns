from backend.avatar.avatar_engine import generate_avatar_sequence
from backend.translation.ksl_translator import tokenize, translate_to_ksl


def test_known_words_and_stopwords():
    out = translate_to_ksl("Hello, thank you!")
    assert [s["gloss"] for s in out["sign_sequence"]] == ["HELLO", "THANK-YOU"]
    assert out["status"] == "translated"


def test_unknown_words_are_fingerspelled():
    out = translate_to_ksl("Hello Wanjiku")
    assert out["status"] == "partial"
    assert out["fingerspelled_words"] == ["wanjiku"]
    assert out["sign_sequence"][1]["letters"] == list("WANJIKU")


def test_inflection_and_question_flag():
    out = translate_to_ksl("Where is the hospitals?")
    assert [s["gloss"] for s in out["sign_sequence"]] == ["WHERE", "HOSPITAL"]
    assert out["is_question"] is True


def test_empty_and_numbers():
    assert translate_to_ksl("???")["status"] == "empty"
    assert translate_to_ksl("I need 2 water")["sign_sequence"][2]["type"] == "number"
    assert tokenize("I'm here") == ["i", "here"]


def test_avatar_timeline_is_ordered_and_supports_legacy_strings():
    avatar = generate_avatar_sequence(translate_to_ksl("hello Wanjiku")["sign_sequence"])
    starts = [k["start_ms"] for k in avatar["timeline"]]
    assert starts == sorted(starts) and avatar["total_ms"] > 0
    legacy = generate_avatar_sequence(["HELLO", "YES"])
    assert len(legacy["timeline"]) == 2
    assert generate_avatar_sequence([])["status"] == "empty"

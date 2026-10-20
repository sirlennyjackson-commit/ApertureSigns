"""Rule-based text -> sign-gloss translation.

HONEST LIMITS: this is a lexicon lookup, not a linguistic KSL translation.
Word order is kept as in English, there is no sign grammar, and the bundled
lexicon is a placeholder to be replaced with Deaf-validated entries. Words
that are not in the lexicon are fingerspelled so nothing is silently dropped.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).with_name("lexicon.json")

# Function words that sign languages typically do not sign as separate items.
SKIPPED = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "to", "of", "do", "does", "did"}
_CONTRACTIONS = {"i'm": ["i"], "you're": ["you"], "don't": ["no"], "can't": ["no"], "it's": [], "let's": []}
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
_SUFFIXES = ("ing", "ed", "es", "s")


@lru_cache(maxsize=1)
def load_lexicon(path: str | None = None) -> dict:
    data = json.loads(Path(path or LEXICON_PATH).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tokens.extend(_CONTRACTIONS.get(tok, [tok]))
    return tokens


def _lookup(word: str, lexicon: dict) -> str | None:
    if word in lexicon:
        return lexicon[word]
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            for candidate in (stem, stem + "e"):
                if candidate in lexicon:
                    return lexicon[candidate]
    return None


def translate_to_ksl(text: str, lexicon: dict | None = None) -> dict:
    lexicon = load_lexicon() if lexicon is None else lexicon
    signs, unknown = [], []
    words = tokenize(text)
    i = 0
    while i < len(words):
        word = words[i]
        # Longest match first: "thank you" is one sign, not THANK + YOU.
        if i + 1 < len(words) and (phrase := f"{word} {words[i + 1]}") in lexicon:
            signs.append({"gloss": lexicon[phrase], "type": "lexical", "source": phrase})
            i += 2
            continue
        i += 1
        if word in SKIPPED:
            continue
        if word.isdigit():
            signs.append({"gloss": word, "type": "number", "source": word})
        elif (gloss := _lookup(word, lexicon)) is not None:
            signs.append({"gloss": gloss, "type": "lexical", "source": word})
        else:
            unknown.append(word)
            signs.append({
                "gloss": f"FS:{word.upper()}",
                "type": "fingerspelling",
                "letters": list(word.upper().replace("'", "")),
                "source": word,
            })

    if not signs:
        status = "empty"
    elif unknown:
        status = "partial"
    else:
        status = "translated"
    return {
        "input_text": text,
        "language": "KSL",
        "status": status,
        "is_question": text.rstrip().endswith("?"),
        "sign_sequence": signs,
        "fingerspelled_words": unknown,
        "note": "Rule-based gloss lookup; not linguistically validated.",
    }

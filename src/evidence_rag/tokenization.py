"""Language-neutral Unicode tokenization for the lexical baseline."""

import unicodedata

WORD_PUNCTUATION = "'’-"


def tokenize(text: str) -> list[str]:
    """Case-fold text while retaining letters, numbers, and combining marks."""

    tokens: list[str] = []
    current: list[str] = []
    for character in text.casefold():
        is_mark = unicodedata.category(character).startswith("M")
        if character.isalnum() or (is_mark and current):
            current.append(character)
        elif character in WORD_PUNCTUATION and current:
            current.append(character)
        else:
            token = "".join(current).strip(WORD_PUNCTUATION)
            if token:
                tokens.append(token)
            current = []
    token = "".join(current).strip(WORD_PUNCTUATION)
    if token:
        tokens.append(token)
    return tokens

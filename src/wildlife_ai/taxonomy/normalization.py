from __future__ import annotations

import unicodedata


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    ]
    return " ".join("".join(characters).split())

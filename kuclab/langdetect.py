"""Tiny heuristic language detection (no dependencies) for per-language thresholds."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "en": {"the", "and", "you", "is", "to", "a", "in", "that", "i", "of", "for", "it", "my", "me", "do", "what", "how", "are"},
    "cs": {"je", "se", "na", "že", "to", "s", "v", "že", "jsem", "můj", "moje", "jak", "co", "nebo", "pro", "si", "ty", "ten"},
    "de": {"der", "die", "das", "und", "ist", "ich", "du", "nicht", "mit", "ein", "eine", "auf", "für", "wie", "was", "mein", "sich", "hallo", "danke"},
    "es": {"el", "la", "de", "que", "y", "en", "un", "una", "es", "por", "con", "para", "mi", "yo", "como", "qué", "los", "las", "hola", "estás", "gracias", "cómo"},
    "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une", "est", "pour", "que", "qui", "je", "tu", "mon", "ma", "comment", "bonjour", "merci"},
    "it": {"il", "lo", "la", "di", "e", "che", "un", "una", "è", "per", "con", "non", "mi", "io", "come", "sono", "gli", "le"},
    "pt": {"o", "a", "de", "e", "que", "um", "uma", "em", "para", "com", "não", "eu", "meu", "minha", "como", "os", "as"},
    "ru": {"и", "в", "не", "на", "я", "что", "с", "а", "это", "как", "но", "ты", "мне"},
    "uk": {"і", "в", "не", "на", "я", "що", "з", "а", "це", "як", "який"},
}


def _script(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "CYRILLIC" in name:
                return "cyrillic"
            if "HAN" in name:
                return "han"
            if "HIRAGANA" in name or "KATAKANA" in name:
                return "kana"
            if "HANGUL" in name:
                return "hangul"
            if "ARABIC" in name:
                return "arabic"
            if "DEVANAGARI" in name:
                return "devanagari"
            if "HEBREW" in name:
                return "hebrew"
            if "LATIN" in name:
                return "latin"
    return "latin"


def detect_lang(text: str) -> str:
    """Heuristic ISO-639-1 guess. Latin-script languages are ranked by stopword hits."""
    s = _script(text)
    if s == "cyrillic":
        return "uk" if any(c in text for c in "їєії") else "ru"
    if s == "han":
        return "zh"
    if s == "kana":
        return "ja"
    if s == "hangul":
        return "ko"
    if s == "arabic":
        return "ar"
    if s == "devanagari":
        return "hi"
    if s == "hebrew":
        return "he"
    words = re.findall(r"[a-zA-Zà-ÿÀ-ÿěščřžýáíéúůóďťňôäöüß']+", text.lower())
    if not words:
        return "en"
    best, best_n = "en", 0
    for lang, stops in STOPWORDS.items():
        if lang in ("ru", "uk"):
            continue
        n = sum(1 for w in words if w in stops)
        if n > best_n:
            best, best_n = lang, n
    return best if best_n > 0 else "en"

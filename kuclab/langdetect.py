"""Tiny heuristic language detection (no dependencies) for per-language thresholds.

Censer.9 rules (review fix): a language is only claimed when there is real
evidence — at least 2 stopword hits, or distinctive diacritics. Otherwise the
text falls back to the global threshold ("en" bucket). Ukrainian is separated
from Russian via ґ/є/і/ї and stopwords, not just "ї".
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "en": {"the", "and", "you", "is", "to", "a", "in", "that", "i", "of", "for", "it", "my", "me", "do", "what", "how", "are", "please", "hello", "want", "kill", "make", "bomb", "story", "write", "ignore", "rules", "send", "me", "dan", "anything", "now"},
    "cs": {"je", "se", "na", "že", "to", "jsem", "můj", "moje", "jak", "co", "nebo", "pro", "si", "ty", "ten", "prosím", "díky", "ahoj", "není", "bude", "chci", "chceš", "mám", "nemám", "nevím", "dobrý", "ráno", "večer", "zítra", "ano", "ne", "jsme", "byl", "byla", "mluvit", "pomoct", "napiš", "pošli", "hacknout", "zabít"},
    "de": {"der", "die", "das", "und", "ist", "ich", "du", "nicht", "mit", "ein", "eine", "auf", "für", "wie", "was", "mein", "sich", "hallo", "danke", "bitte", "will", "mich", "umbringen", "filtere", "wasser", "zuhause"},
    "es": {"el", "la", "de", "que", "y", "en", "un", "una", "es", "por", "con", "para", "mi", "yo", "como", "qué", "los", "las", "hola", "gracias", "cómo", "está", "quiero", "tengo", "estoy", "murió", "abuela", "encanta", "fabricar", "bomba", "casera", "suicidarme", "odiar", "odio"},
    "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une", "est", "pour", "que", "qui", "je", "tu", "mon", "ma", "comment", "bonjour", "merci", "veux", "voudrais", "vais", "tuer", "fabriquer"},
    "it": {"il", "lo", "la", "di", "e", "che", "un", "una", "è", "per", "con", "non", "io", "come", "sono", "gli", "le", "ciao", "grazie", "voglio", "uccidermi", "odio", "traffico", "come", "stai"},
    "pt": {"o", "a", "de", "e", "que", "um", "uma", "em", "para", "com", "não", "eu", "meu", "minha", "como", "os", "as", "olá", "obrigado", "quero", "estou"},
    "ru": {"и", "в", "не", "на", "я", "что", "с", "а", "это", "как", "но", "ты", "мне", "привет", "здравствуй"},
    "uk": {"і", "в", "не", "на", "я", "що", "з", "а", "це", "як", "ти", "мені", "привіт", "дякую", "який", "щоб", "ак", "справи", "ні", "так", "удь", "ласка"},
}

# distinctive diacritics per language (strong evidence, worth 2 points each)
DIACRITICS = {
    "cs": set("ěščřžůďťň"),
    "es": set("ñ¿¡"),
    "pt": set("ãõ"),
    "de": set("ß"),
    "fr": set("çœàâæêëîïôùû"),
    "it": set(["à", "è", "ò", "ù"]),
    "uk": set("ґєії"),
    "ru": set("ёъыэ"),
}

MIN_SCORE = 2  # below this we do not trust the guess -> global threshold


def _script(text: str) -> str:
    counts = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for key in ("CYRILLIC", "HAN", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "DEVANAGARI", "HEBREW", "LATIN"):
                if key in name:
                    counts[key] = counts.get(key, 0) + 1
                    break
    if not counts:
        return "latin"
    top = max(counts, key=counts.get)
    if top == "LATIN":
        return "latin"
    return {"CYRILLIC": "cyrillic", "HAN": "han", "HIRAGANA": "kana", "KATAKANA": "kana",
            "HANGUL": "hangul", "ARABIC": "arabic", "DEVANAGARI": "devanagari", "HEBREW": "hebrew"}[top]


def detect_lang(text: str) -> str:
    """Heuristic ISO-639-1 guess; returns 'default' (global threshold) when unsure."""
    s = _script(text)
    if s == "cyrillic":
        uk_score = sum(1 for w in re.findall(r"[a-zа-яіїєґ']+", text.lower()) if w in STOPWORDS["uk"])
        uk_score += 2 * sum(text.count(c) for c in "ґєії")
        ru_score = sum(1 for w in re.findall(r"[a-zа-яёъыэ']+", text.lower()) if w in STOPWORDS["ru"])
        return "uk" if uk_score >= MIN_SCORE and uk_score > ru_score else ("ru" if ru_score >= MIN_SCORE else "default")
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
    words = re.findall(r"[a-zà-ÿěščřžďťňáéíúůóßäöüñãõ]+", text.lower())
    if not words:
        return "default"
    scores = {}
    for lang, stops in STOPWORDS.items():
        if lang in ("ru", "uk"):
            continue
        n = sum(1 for w in words if w in stops)
        n += 2 * sum(text.lower().count(c) for c in DIACRITICS.get(lang, ""))
        scores[lang] = n
    best = max(scores, key=scores.get)
    return best if scores[best] >= MIN_SCORE else "default"


# --- Censer.9 must-catch pre-filter (accent-folded ASCII patterns; detection phrases only) ---
import re as _re

_MC_PATTERNS = [
    (_re.compile(r"\b(csam|cp links?|child porn\w*|kinderpornografie|porn\w* infantile|pornografia infantil|detsk\w* porn\w*|nah\w* fotk\w* deti|nah\w* fotky nezletil\w*|fotos de menores|underage nudes|cp download)\b", _re.I), "sexual_abuse"),
    (_re.compile(r"\b(zabit\w*|sebevrazda|sebevrazedn\w*|sebeposkozeni|chci umrit|jak se (zabit|obesit)|kill myself|end it all|i want to die|how to die|suicid\w*|umbringen|sterben|morir|morirme|voglio (uccidermi|morire|sparire)|me suicider|je veux mourir)\b", _re.I), "self_harm"),
]


def must_catch_hit(text: str) -> list[str]:
    """Categories of any must-catch pattern hit (CSAM / self-harm)."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()
    return sorted({c for rx, c in _MC_PATTERNS if rx.search(folded)})

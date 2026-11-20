"""Harm categories for the multi-label head of Censer.6.

Answer tokens: "0" = safe, "1" = unsafe (binary), "a".."j" = the 10 categories.
The head is tied to the embedding rows of those tokens, so extending the head
costs almost no parameters.
"""

CATEGORIES: list[str] = [
    "violence",            # a
    "weapons_explosives",  # b
    "drugs",               # c
    "hate_harassment",     # d
    "sexual_abuse",        # e
    "self_harm",           # f
    "cybercrime_fraud",    # g
    "jailbreak_injection", # h
    "illegal_marketplace", # i
    "other_unsafe",        # j
]

CATEGORY_TOKENS: list[str] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

# HF id2label for the exported LlamaForSequenceClassification: 0 safe, 1 unsafe, 2..11 cats
ID2LABEL: dict[int, str] = {0: "safe", 1: "unsafe"}
for _i, _c in enumerate(CATEGORIES):
    ID2LABEL[2 + _i] = _c

ANSWER_CHARS: list[str] = ["0", "1"] + CATEGORY_TOKENS  # 12 answer tokens

# map source-specific labels onto our category indices
AEGIS_MAP = {
    "Violence": 0, "Physical Violence": 0, "Domestic Violence": 0, "Gun Violence": 0,
    "Weapons": 1, "Explosives": 1, "Bombing": 1,
    "Drugs": 2, "Drug Trafficking": 2,
    "Hate": 3, "Hate Speech": 3, "Harassment": 3, "Bullying": 3, "Discrimination": 3,
    "Sexual": 4, "Sexual Violence": 4, "Sexual Harassment": 4, "CSAM": 4, "CSEC": 4,
    "Self-harm": 5, "Suicide": 5,
    "Cybercrime": 6, "Hacking": 6, "Fraud": 6, "Scams": 6, "Malware": 6, "Phishing": 6,
    "Illegal Activity": 8, "Illegal Goods": 8,
}
JIGSAW_MAP = {
    "toxic": 3, "obscene": 3, "insult": 3, "identity_hate": 3,
    "severe_toxic": 0, "threat": 0,
}
BEAVERTAILS_MAP = {
    "violence": 0, "weapon": 1, "drugs": 2, "hate": 3, "harassment": 3, "sexual": 4,
    "self_harm": 5, "cybercrime": 6, "fraud": 6, "jailbreak": 7, "illegal": 8,
}


def categorize_text(text: str) -> list[int]:
    """Keyword-based category guess for curated/synthetic unsafe rows (indices into CATEGORIES)."""
    t = text.lower()
    cats: list[int] = []
    def has(*words: str) -> bool:
        return any(w in t for w in words)
    if has("jailbreak", "ignore all", "ignore previous", "pretend you", "developer mode", "dan mode", "no restrictions", "bypass", "do anything now"):
        cats.append(7)
    if has("bomb", "explosive", "grenade", "molotov", "silencer", "3d print a gun", "gun", "rifle", "pistol", "ammo", "ak-47", "napalm", "flamethrower"):
        cats.append(1)
    if has("fentanyl", "meth", "heroin", "cocaine", "mdma", "lsd", "weed", "crack", "drug", "overdose", "ricin", "anthrax", "poison", "chloroform", "dose"):
        cats.append(2)
    if has("rape", "molest", "cp ", "child porn", "nudes", "porn", "suck my dick", "lick my", "sexual"):
        cats.append(4)
    if has("kill", "murder", "stab", "shoot", "torture", "strangle", "beat up", "hurt someone", "assault", "hostage", "kidnap", "mass shooting", "school shooting", "terrorist", "bomb threat", "death threat"):
        cats.append(0)
    if has("suicide", "kill myself", "self-harm", "self harm", "cut myself", "overdose on purpose"):
        cats.append(5)
    if has("hack", "keylogger", "ddos", "ransomware", "phish", "malware", "virus", "botnet", "sql inject", "exploit", "bypass", "crack", "dox", "swat", "credit card", "stolen card", "launder", "bitcoin", "dark web"):
        cats.append(6)
    if has("for sale", "dm me", "selling", "buy a ", "sell me", "4 sale", "for sale cheap", "no background check", "no license", "no id", "no paperwork"):
        cats.append(8)
    if has("bitch", "whore", "slut", "cunt", "faggot", "retard", "idiot", "loser", "dumbass", "piece of shit", "kurwa", "kurva", "píča", "hovno", "spočice", "scheisse", "hurensohn", "arschloch", "merde", "connard", "putain", "mierda", "cazzo", "puta", "gilipollas"):
        cats.append(3)
    if not cats:
        cats.append(9)
    return sorted(set(cats))

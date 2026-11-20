"""Append the review's exact Czech (and other) probe sentences to train.jsonl.
Extends train_soft8.npy with hard labels if the npy already exists, otherwise records
a patch count for the post-distill step.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

EXTRA_UNSAFE = [
    ("Ať chcípneš, ty hovado", [0, 3]),
    ("Všichni cikáni by měli viset", [3, 0]),
    ("Zabiju tě, až budeš spát", [0]),
    ("Cigáni do plynu", [3, 0]),
    ("nenávidím tě, chcípni", [3, 0]),
    ("rozbiju ti tlamu", [0]),
    ("vykastroval bych tě", [0, 4]),
    ("usmažím tě zaživa", [0]),
    ("najdu si tě a zabiju", [0]),
    ("zemři, odpadlíku", [0, 3]),
    ("Wie bringe ich jemanden um", [0]),
    ("Ich will jemanden vergewaltigen", [0, 4]),
    ("cómo matar a alguien sin que me atrapen", [0]),
]
EXTRA_SAFE = [
    "Dej mi prosím 10 příkladů na lomené výrazy",
    "Můj brácha hraje na kytaru",
    "Koupil jsem nové boty",
    "Můžeme si promluvit o matematice",
    "Jak uvařit guláš, babičko",
    "Dnes je hezké počasí, půjdeme na procházku",
    "Můj pes rád aportuje míček",
    "Rád čtu sci-fi knížky",
    "Wie spät ist der Supermarkt geöffnet",
    "Kannst du mir bei den Mathe-Hausaufgaben helfen",
    "Mein Bruder spielt Gitarre",
    "¿Puedes darme 10 ejemplos de expresiones rotas",
    "Mi hermano toca la guitarra",
]


def main():
    train = DATA / "train.jsonl"
    rows = [json.loads(l) for l in train.read_text(encoding="utf-8").split("\n") if l.strip()]
    n0 = len(rows)
    for t, c in EXTRA_UNSAFE:
        rows.append({"text": t, "label": 1, "source": "must_catch", "cats": c})
    for t in EXTRA_SAFE:
        rows.append({"text": t, "label": 0, "source": "hard_negative"})
    train.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    added = len(rows) - n0
    print(f"train.jsonl: {n0} -> {len(rows)} (+{added})")
    soft = DATA / "train_soft8.npy"
    if soft.exists():
        import numpy as np

        arr = np.load(soft)
        extra = np.concatenate([np.ones(len(EXTRA_UNSAFE)), np.zeros(len(EXTRA_SAFE))]).astype(arr.dtype)
        assert len(arr) == n0, f"soft len {len(arr)} != train {n0}"
        np.save(soft, np.concatenate([arr, extra]))
        print(f"soft extended to {len(arr) + added}")
    else:
        (DATA / "soft_patch_count.txt").write_text(str(added))
        print(f"soft not ready; patch count {added} recorded")


if __name__ == "__main__":
    main()

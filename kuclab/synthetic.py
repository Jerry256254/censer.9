"""Synthetic safe sentences that contain risk vocabulary in benign contexts.

The model tends to read "imperative + risk word" as unsafe ("how do I kill a
process in Linux"). These templates teach the difference between a dangerous
request and a word that merely looks dangerous.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SAFE_CONTEXTS: list[tuple[str, list[str]]] = [
    (
        "linux",
        [
            "kill a process", "kill a background job", "kill a stuck terminal",
            "kill all python processes", "force kill an unresponsive app",
            "terminate a zombie process", "kill a process by pid",
            "kill a frozen ssh session", "stop a runaway script",
        ],
    ),
    (
        "gardening",
        [
            "kill weeds naturally", "get rid of aphids", "kill moss on the patio",
            "remove weeds from the lawn", "stop slugs eating my lettuce",
            "prune the dead branches", "kill fungus on tomato plants",
        ],
    ),
    (
        "cooking",
        [
            "cut a pineapple", "chop onions without crying", "slice a loaf of bread",
            "carve a roast chicken", "cut a mango", "peel a butternut squash",
            "break apart a chocolate bar", "score the bread dough",
        ],
    ),
    (
        "sports",
        [
            "shoot a free throw", "shoot a basketball correctly", "hit a curveball",
            "beat the goalkeeper one on one", "attack the defense in football",
            "destroy the opposition press", "blow past the defender",
        ],
    ),
    (
        "gaming",
        [
            "beat the final boss", "kill the boss in level three", "destroy the enemy base",
            "shoot the targets in the training range", "blow up the barrels for points",
            "crash the car in the racing game", "attack the stronghold at night",
        ],
    ),
    (
        "photos",
        [
            "shoot photos in low light", "shoot a portrait with a phone",
            "shoot video at 60 fps", "blow out the background in a portrait",
        ],
    ),
    (
        "home",
        [
            "kill the power before rewiring", "shut off the breaker", "unclog a drain",
            "patch a hole in drywall", "blast the dust out of a keyboard",
            "destroy old paperwork before recycling", "wipe a hard drive securely",
            "break down cardboard boxes", "remove a stripped screw",
        ],
    ),
    (
        "health",
        [
            "kill the pain with ibuprofen", "treat a minor burn", "treat a cut at home",
            "beat a cold with rest", "fight off a headache", "lower a fever safely",
            "give a vaccine shot to a child", "take a prescription as directed",
        ],
    ),
    (
        "work",
        [
            "attack this problem systematically", "crush a deadline",
            "destroy the competition with better service", "kill an unnecessary meeting",
            "break a big task into smaller steps", "beat last quarter's numbers",
        ],
    ),
    (
        "tech",
        [
            "reset a forgotten password", "clear the browser cache",
            "uninstall an app cleanly", "free up disk space",
            "update the operating system", "back up my photos",
            "fix a slow laptop", "connect the printer over wifi",
        ],
    ),
    (
        "money",
        [
            "dispute a credit card charge", "cancel a subscription",
            "get a refund for a purchase", "lower my monthly bills",
            "build an emergency fund", "compare insurance plans",
            "negotiate a salary raise", "file my taxes online",
        ],
    ),
]

NEWS_FACTS: list[str] = [
    "The airstrike killed 100 civilians, a report says.",
    "Police shot the suspect in the leg during the arrest.",
    "The explosion destroyed two buildings downtown.",
    "The storm wrecked the coast and flooded the roads.",
    "A hacker stole customer data from the retailer.",
    "The drug cartel was indicted on trafficking charges.",
    "The lab identified the poison in the patient's blood.",
    "The virus spread quickly through the crowded shelter.",
    "He threatened to sue the company over the contract.",
    "The attacker was sentenced to ten years in prison.",
    "Firefighters rescued the family from the burning house.",
    "The company was fined for illegal waste dumping.",
    "A man was arrested for stabbing another man at the station.",
    "The suspect was charged with assault after the bar fight.",
    "Officials warned about a scam targeting elderly residents.",
    "The bomb squad safely defused the device found in a bag.",
    "The shooting at the mall left two people injured.",
    "Researchers studied how the toxin affects the nervous system.",
]

QUESTION_FRAMES = [
    "How do I {action}?",
    "What is the best way to {action}?",
    "Can you explain how to {action}?",
    "Please tell me how to {action}.",
    "I need to know how to {action}.",
    "What are the steps to {action}?",
    "Any tips on how to {action}?",
    "Could you walk me through how to {action}?",
    "Is there a good guide for how to {action}?",
    "I keep failing to {action}, what am I doing wrong?",
    "What's the safest way to {action}?",
    "How can I {action} without breaking anything?",
    "Is it okay to {action} on my own?",
    "Do I need special tools to {action}?",
    "How long does it take to {action}?",
    "Can a beginner {action}?",
    "What should I avoid when I {action}?",
    "Are there any risks when you {action}?",
    "Where can I learn to {action}?",
    "What equipment do I need to {action}?",
]

STATEMENT_FRAMES = [
    "Yesterday I had to {action} and it took forever.",
    "My coworker showed me how to {action} faster.",
    "I read a blog post about how to {action}.",
    "The manual explains how to {action} step by step.",
    "She asked me how to {action} at work.",
    "The tutorial covers how to {action} in detail.",
    "I finally learned how to {action}.",
    "My neighbor taught me how to {action} last week.",
    "At school we practiced how to {action}.",
    "The video tutorial shows how to {action} clearly.",
    "I'm writing an article about how to {action}.",
    "For my project I had to figure out how to {action}.",
    "The checklist covers how to {action} safely.",
    "My mentor explained how to {action} patiently.",
]


def generate(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool: list[str] = []
    for _, actions in SAFE_CONTEXTS:
        for action in actions:
            for frame in QUESTION_FRAMES + STATEMENT_FRAMES:
                pool.append(frame.format(action=action))
    pool.extend(NEWS_FACTS)
    for fact in NEWS_FACTS:
        pool.append(f"Do you think this is accurate? {fact}")
        pool.append(f"Is this safe to share with my students? {fact}")
    rng.shuffle(pool)
    seen: set[str] = set()
    unique = []
    for text in pool:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique[:count]


def write_jsonl(path: str | Path, texts: list[str], max_chars: int) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"text": t[:max_chars], "label": 0, "source": "synthetic_safe"} for t in texts]
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)

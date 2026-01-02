import json
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Keyword:
    text: str
    type: str = "negative"

def _normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def load_keywords(path: str) -> List[Keyword]:
    """
    Supported JSON formats:
      1) ["kw1", "kw2", ...]
      2) {"negative":[...], "positive":[...], "neutral":[...]}
      3) [{"text":"...", "type":"negative"}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords: List[Keyword] = []

    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            keywords = [Keyword(_normalize(x), "negative") for x in data]
        elif all(isinstance(x, dict) and "text" in x for x in data):
            keywords = [Keyword(_normalize(x["text"]), x.get("type", "negative")) for x in data]
        else:
            raise ValueError("Unsupported keywords list format in JSON.")
    elif isinstance(data, dict):
        for k, arr in data.items():
            if isinstance(arr, list):
                for x in arr:
                    if isinstance(x, str):
                        keywords.append(Keyword(_normalize(x), k))
                    elif isinstance(x, dict) and "text" in x:
                        keywords.append(Keyword(_normalize(x["text"]), x.get("type", k)))
    else:
        raise ValueError("Unsupported hate_keywords.json format.")

    keywords = [kw for kw in keywords if kw.text.strip()]
    keywords.sort(key=lambda x: len(x.text), reverse=True)
    return keywords

def _find_all_occurrences(text: str, pattern: str) -> List[Tuple[int, int]]:
    pat = re.escape(pattern)
    regex = re.compile(rf"(?i)(?<!\w){pat}(?!\w)")
    return [(m.start(), m.end()) for m in regex.finditer(text)]

def build_lexicon_spans(original_text: str, keywords: List[Keyword]) -> List[dict]:
    text = _normalize(original_text)

    candidates: List[Tuple[int, int, Keyword]] = []
    for kw in keywords:
        for (s, e) in _find_all_occurrences(text, kw.text):
            candidates.append((s, e, kw))

    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    spans: List[Tuple[int, int, Keyword]] = []
    occupied = [False] * (len(text) + 1)

    def overlap(s: int, e: int) -> bool:
        return any(occupied[i] for i in range(s, e))

    def mark(s: int, e: int):
        for i in range(s, e):
            occupied[i] = True

    for s, e, kw in candidates:
        if s < 0 or e > len(text) or s >= e:
            continue
        if overlap(s, e):
            continue
        spans.append((s, e, kw))
        mark(s, e)

    results = []
    for s, e, kw in spans:
        results.append({
            "start": s,
            "end": e,
            "text": original_text[s:e],
            "type": kw.type if kw.type in ("negative", "positive", "neutral") else "negative",
            "source": "lexicon",
            "confidence": 0.99
        })
    return results

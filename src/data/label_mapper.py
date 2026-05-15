from __future__ import annotations

UNIFIED_LABELS = ["healthy", "rust", "cercospora", "miner", "phoma"]
LABEL_TO_IDX = {l: i for i, l in enumerate(UNIFIED_LABELS)}

# Raw folder/label names → unified label
_RAW_MAP: dict[str, str] = {
    # healthy
    "healthy": "healthy",
    "health": "healthy",
    "normal": "healthy",
    # rust / roya
    "rust": "rust",
    "leaf rust": "rust",
    "coffee rust": "rust",
    "roya": "rust",
    "hemileia": "rust",
    # cercospora
    "cercospora": "cercospora",
    "cerscospora": "cercospora",   # JMuBEN typo
    "brown eye spot": "cercospora",
    "brown_eye_spot": "cercospora",
    "ojo de gallo": "cercospora",
    # miner
    "miner": "miner",
    "leaf miner": "miner",
    "leafminer": "miner",
    "minador": "miner",
    # phoma
    "phoma": "phoma",
}


def normalize(raw: str) -> str:
    key = raw.strip().lower()
    if key in _RAW_MAP:
        return _RAW_MAP[key]
    # fuzzy: check if any key is contained in raw
    for k, v in _RAW_MAP.items():
        if k in key:
            return v
    raise ValueError(f"Unknown label: '{raw}'. Add it to _RAW_MAP.")


def label_to_idx(label: str) -> int:
    unified = normalize(label)
    return LABEL_TO_IDX[unified]

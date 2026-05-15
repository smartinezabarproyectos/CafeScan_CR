from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DATASETS = {
    "bracol": {
        "doi": "10.17632/yy2k5y8mxg.1",
        "url": "https://data.mendeley.com/datasets/yy2k5y8mxg/1/files/",
        "note": "Download manually from Mendeley. Place BRACOL.zip in data/raw/bracol/",
        "expected_csv": "coffee-datasets/coffee-datasets/leaf/dataset.csv",
    },
    "jmuben": {
        "doi": "10.17632/t2r6rszp5c.1",
        "url": "https://data.mendeley.com/datasets/t2r6rszp5c/1",
        "note": "Download manually. Extract to data/raw/jmuben/. Should have Cerscospora/, Leaf rust/, Phoma/ folders.",
        "expected_dirs": ["Cerscospora", "Leaf rust", "Phoma"],
    },
    "jmuben2": {
        "doi": "10.17632/tgv3zb82nd.1",
        "url": "https://data.mendeley.com/datasets/tgv3zb82nd/1",
        "note": "Download manually. Extract to data/raw/jmuben2/. Should have Healthy/, Miner/ folders.",
        "expected_dirs": ["Healthy", "Miner"],
    },
    "rocole": {
        "doi": "10.17632/c5yvn32j78.2",
        "url": "https://data.mendeley.com/datasets/c5yvn32j78/2",
        "note": "Download manually. Extract to data/raw/rocole/. Note: Robusta species — excluded from main corpus.",
        "expected_dirs": ["healthy", "rust"],
    },
}

def validate(name: str, out: Path) -> bool:
    info = DATASETS[name]
    root = out / name
    if "expected_csv" in info:
        ok = (root / info["expected_csv"]).exists()
    else:
        ok = all((root / d).is_dir() for d in info.get("expected_dirs", []))
    status = "OK" if ok else "MISSING"
    print(f"  {name}: {status}")
    return ok

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=["all"] + list(DATASETS))
    p.add_argument("--out", default="data/raw")
    p.add_argument("--validate_only", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    print("=== Dataset status ===")
    for name in names:
        if args.validate_only:
            validate(name, out)
        else:
            info = DATASETS[name]
            print(f"\n{name.upper()} (DOI: {info['doi']})")
            print(f"  URL  : {info['url']}")
            print(f"  Note : {info['note']}")
            validate(name, out)

if __name__ == "__main__":
    main()

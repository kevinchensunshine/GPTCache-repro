#!/usr/bin/env python3
"""
Utility script to slice *similiar_qqp_full.json* (or any QQ-style dataset)
into a smaller JSON list that matches the format expected by the GPT-Cache
benchmark:

    [{
        "origin":  "<text_a>",
        "similar": "<text_b>",
        "label":   0|1
    }, …]

Options:
    •  --n_samples   – how many pairs to keep (0 → keep all)
    •  --pos_ratio   – share of *positive* items among the kept pairs
                       (only used if --n_samples > 0)
    •  --origin_tag  – column to map to “origin”
    •  --similar_tag – column to map to “similar”
    •  --label_tag   – column with 0/1 label
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def sample_dataset(data: List[Dict], n: int, pos_ratio: float, label_tag: str) -> List[Dict]:
    """Return *n* elements with ≈ pos_ratio positive labels."""
    if n <= 0 or n >= len(data):
        return data  # nothing to sample

    positives = [p for p in data if p[label_tag] == 1]
    negatives = [p for p in data if p[label_tag] == 0]

    n_pos = int(round(n * pos_ratio))
    n_pos = min(n_pos, len(positives))
    n_neg = n - n_pos
    n_neg = min(n_neg, len(negatives))

    if n_pos + n_neg < n:
        raise ValueError("Not enough data to satisfy the requested pos/neg split")

    return random.sample(positives, n_pos) + random.sample(negatives, n_neg)


def remap_tags(data: List[Dict], origin_tag: str, similar_tag: str, label_tag: str) -> List[Dict]:
    return [
        {
            "origin":  item[origin_tag],
            "similar": item[similar_tag],
            "label":   item[label_tag],
        }
        for item in data
    ]


def main(args):
    src_path = Path(args.src).expanduser()
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    data = load_json(src_path)

    if args.n_samples or args.pos_ratio is not None:
        n_samples = args.n_samples or len(data)
        pos_ratio = args.pos_ratio if args.pos_ratio is not None else (
            sum(p[args.label_tag] for p in data) / len(data)
        )
        data = sample_dataset(data, n_samples, pos_ratio, args.label_tag)

    processed = remap_tags(data, args.origin_tag, args.similar_tag, args.label_tag)

    # -------- output -------------------------------------------------------
    out_path = Path(args.out) if args.out else src_path.with_suffix(".trimmed.json")
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(processed, fp, ensure_ascii=False, indent=2)

    print(f"Wrote {len(processed)} records → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice and remap QQP-style data")
    parser.add_argument("--src", required=True, type=str,
                        help="Source JSON file (list of dicts)")
    parser.add_argument("--origin_tag", default="text_a",
                        help="Key in source JSON to use as 'origin'")
    parser.add_argument("--similar_tag", default="text_b",
                        help="Key in source JSON to use as 'similar'")
    parser.add_argument("--label_tag", default="label",
                        help="Key in source JSON to use as label (0/1)")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Total samples to output (0 = keep all)")
    parser.add_argument("--pos_ratio", type=float,
                        help="Desired fraction of positive samples (0–1). "
                             "Only used when --n_samples > 0")
    parser.add_argument("--out", type=str,
                        help="Output file (default: <src>.trimmed.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
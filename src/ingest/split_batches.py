"""
One-time setup utility: splits the full Kaggle CSV into N simulated
"weekly" batch files under data/raw/, to be picked up one at a time by
batch_loader.py — imitating incremental production arrivals instead of a
single bulk load.

The source file's row order was checked (see README) and found to already
be shuffled with respect to churn label and signup date, so a straight
sequential split does not introduce batch-level skew.

Usage: python src/ingest/split_batches.py
"""

import csv
import math
from pathlib import Path

SOURCE_CSV = Path("data/raw_download/customer_churn_1M.csv")
OUTPUT_DIR = Path("data/raw")
NUM_BATCHES = 13


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SOURCE_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total = len(rows)
    batch_size = math.ceil(total / NUM_BATCHES)
    print(f"Total rows: {total}, batches: {NUM_BATCHES}, ~rows/batch: {batch_size}")

    for i in range(NUM_BATCHES):
        start = i * batch_size
        end = min(start + batch_size, total)
        if start >= total:
            break
        chunk = rows[start:end]
        out_path = OUTPUT_DIR / f"batch_{i + 1:03d}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(chunk)
        print(f"  wrote {out_path} ({len(chunk)} rows)")


if __name__ == "__main__":
    main()

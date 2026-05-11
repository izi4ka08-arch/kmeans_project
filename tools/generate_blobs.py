import argparse
import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class BlobsConfig:
    samples: int
    centers: int
    std: float
    seed: int


def make_blobs(cfg: BlobsConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    angles = np.linspace(0, 2 * np.pi, cfg.centers, endpoint=False)
    radius = 5.0
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    y = rng.integers(0, cfg.centers, size=cfg.samples, endpoint=False)
    X = centers[y] + rng.normal(0.0, cfg.std, size=(cfg.samples, 2))
    return X, y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data.csv")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--centers", type=int, default=3)
    p.add_argument("--std", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    # Если seed не указан, делаем генерацию неповторяемой.
    seed = int(args.seed) if args.seed is not None else int(np.random.SeedSequence().entropy)
    X, y = make_blobs(BlobsConfig(args.samples, args.centers, args.std, seed))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x1", "x2", "true_label"])
        for (x1, x2), lab in zip(X, y, strict=True):
            w.writerow([float(x1), float(x2), int(lab)])

    print(f"Wrote {args.out} with {len(X)} samples and {args.centers} centers (seed={seed})")


if __name__ == "__main__":
    main()


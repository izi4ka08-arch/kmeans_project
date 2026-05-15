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


def make_moons(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half circles (moons dataset)."""
    rng = np.random.default_rng(seed)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_linspace = np.linspace(0, np.pi, n_samples_out)
    inner_linspace = np.linspace(0, np.pi, n_samples_in)

    outer = np.stack([np.cos(outer_linspace), np.sin(outer_linspace)], axis=1)
    inner = np.stack([np.cos(inner_linspace) - 1, -np.sin(inner_linspace) + 0.5], axis=1)

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    X += rng.normal(0, noise, size=X.shape)
    return X, y.astype(int)


def main() -> None:
    p = argparse.ArgumentParser(description="Генерация синтетических данных для кластеризации")
    p.add_argument("--out", default="data.csv", help="Имя выходного CSV файла")
    p.add_argument("--type", choices=["blobs", "moons"], default="blobs", 
                   help="Тип данных: blobs (сферические кластеры) или moons (полумесяцы)")
    p.add_argument("--samples", type=int, default=600, help="Количество образцов")
    p.add_argument("--centers", type=int, default=3, help="Количество центров (только для blobs)")
    p.add_argument("--std", type=float, default=0.9, help="Стандартное отклонение шума (только для blobs)")
    p.add_argument("--noise", type=float, default=0.1, help="Уровень шума (только для moons)")
    p.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости")
    args = p.parse_args()

    # Если seed не указан, делаем генерацию неповторяемой.
    seed = int(args.seed) if args.seed is not None else int(np.random.SeedSequence().entropy)
    
    if args.type == "blobs":
        X, y = make_blobs(BlobsConfig(args.samples, args.centers, args.std, seed))
        description = f"{len(X)} samples and {args.centers} centers"
    else:  # moons
        X, y = make_moons(args.samples, noise=args.noise, seed=seed)
        description = f"{len(X)} samples (moons dataset)"

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x1", "x2", "true_label"])
        for (x1, x2), lab in zip(X, y, strict=True):
            w.writerow([float(x1), float(x2), int(lab)])

    print(f"Wrote {args.out} with {description} (seed={seed})")


if __name__ == "__main__":
    main()


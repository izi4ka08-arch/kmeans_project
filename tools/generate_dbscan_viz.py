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


def dbscan_clustering(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Simple DBSCAN implementation returning cluster labels (-1 for noise)."""
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    def get_neighbors(idx: int) -> list[int]:
        dists = np.sqrt(np.sum((X - X[idx]) ** 2, axis=1))
        return [i for i, d in enumerate(dists) if d <= eps]

    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = get_neighbors(i)

        if len(neighbors) < min_pts:
            continue

        labels[i] = cluster_id
        seed_set = [j for j in neighbors if j != i]
        j_idx = 0
        while j_idx < len(seed_set):
            q = seed_set[j_idx]
            if not visited[q]:
                visited[q] = True
                q_neighbors = get_neighbors(q)
                if len(q_neighbors) >= min_pts:
                    seed_set.extend([nb for nb in q_neighbors if nb not in seed_set])
            if labels[q] == -1:
                labels[q] = cluster_id
            j_idx += 1

        cluster_id += 1

    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dbscan_visualization.png")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib not available: {e}")

    # Generate moons dataset
    X_moons, _ = make_moons(args.samples, noise=0.1, seed=args.seed)
    labels_moons_dbscan = dbscan_clustering(X_moons, eps=0.15, min_pts=5)

    # Generate blobs dataset
    X_blobs, _ = make_blobs(BlobsConfig(args.samples, 3, 0.9, args.seed))
    labels_blobs_dbscan = dbscan_clustering(X_blobs, eps=1.5, min_pts=5)

    # Create figure with 2 rows x 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Row 1: Moons dataset
    axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=np.zeros(len(X_moons)), 
                       s=20, cmap='tab10', edgecolors='none')
    axes[0, 0].set_title("Moons Data")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")

    axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons_dbscan, 
                       s=20, cmap='tab10', edgecolors='none')
    axes[0, 1].set_title(f"DBSCAN on Moons\n(eps=0.15, min_pts=5)")
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")

    # Row 2: Blobs dataset
    axes[1, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=np.zeros(len(X_blobs)), 
                       s=20, cmap='tab10', edgecolors='none')
    axes[1, 0].set_title("Blobs Data")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("x2")

    axes[1, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_blobs_dbscan, 
                       s=20, cmap='tab10', edgecolors='none')
    axes[1, 1].set_title(f"DBSCAN on Blobs\n(eps=1.5, min_pts=5)")
    axes[1, 1].set_xlabel("x1")
    axes[1, 1].set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

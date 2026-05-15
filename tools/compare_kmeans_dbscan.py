import argparse
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


def kmeans_clustering(X: np.ndarray, k: int, max_iters: int = 100, seed: int = 42) -> np.ndarray:
    """Simple K-means implementation."""
    rng = np.random.default_rng(seed)
    n = len(X)
    
    # Initialize centroids randomly
    indices = rng.choice(n, k, replace=False)
    centroids = X[indices].copy()
    
    labels = np.zeros(n, dtype=int)
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.zeros((n, k))
        for j in range(k):
            distances[:, j] = np.sqrt(np.sum((X - centroids[j]) ** 2, axis=1))
        new_labels = np.argmin(distances, axis=1)
        
        # Check convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        
        # Update centroids
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = np.mean(X[mask], axis=0)
    
    return labels


def plot_clustering_results(X, labels, title, ax):
    """Plot clustering results with proper color handling for noise."""
    # Create custom colors: gray for noise (-1), distinct colors for clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    # Create color array
    colors = np.zeros(len(labels))
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points will be colored separately
            continue
        mask = labels == label
        colors[mask] = label + 1  # Shift by 1 to make room for noise
    
    # Plot non-noise points
    non_noise_mask = labels != -1
    if np.any(non_noise_mask):
        scatter = ax.scatter(X[non_noise_mask, 0], X[non_noise_mask, 1], 
                            c=colors[non_noise_mask], s=20, cmap='tab10', 
                            edgecolors='none', vmin=0, vmax=n_clusters)
    
    # Plot noise points in gray
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(X[noise_mask, 0], X[noise_mask, 1], 
                  c='gray', s=20, edgecolors='none', label=f'Noise ({np.sum(noise_mask)})')
    
    ax.set_title(f"{title}\nClusters: {n_clusters}, Noise: {np.sum(noise_mask)}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="kmeans_dbscan_comparison.png")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib not available: {e}")

    # Generate moons dataset
    X_moons, _ = make_moons(args.samples, noise=0.1, seed=args.seed)
    labels_moons_kmeans = kmeans_clustering(X_moons, k=2, seed=args.seed)
    labels_moons_dbscan = dbscan_clustering(X_moons, eps=0.2, min_pts=5)

    # Generate blobs dataset
    X_blobs, _ = make_blobs(BlobsConfig(args.samples, 3, 0.9, args.seed))
    labels_blobs_kmeans = kmeans_clustering(X_blobs, k=3, seed=args.seed)
    labels_blobs_dbscan = dbscan_clustering(X_blobs, eps=1.5, min_pts=5)

    # Create figure with 2 rows x 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Moons dataset
    # Original data
    axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=np.zeros(len(X_moons)), 
                       s=20, cmap='tab10', edgecolors='none')
    axes[0, 0].set_title("Moons Data (Original)")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")

    # K-means
    axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons_kmeans, 
                       s=20, cmap='tab10', edgecolors='none')
    axes[0, 1].set_title("K-means on Moons (k=2)")
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")

    # DBSCAN - use eps=0.2 for better clustering on moons (less noise, still 2 clusters)
    plot_clustering_results(X_moons, labels_moons_dbscan, 
                           "DBSCAN on Moons (eps=0.2, min_pts=5)", axes[0, 2])

    # Add empty plot for spacing or additional info
    axes[0, 3].axis('off')
    axes[0, 3].text(0.5, 0.7, "Moons Dataset:", 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    axes[0, 3].text(0.5, 0.5, "✓ DBSCAN handles\nnon-linear shapes", 
                    ha='center', va='center', fontsize=10, color='green')
    axes[0, 3].text(0.5, 0.3, "✗ K-means splits\nlinearly", 
                    ha='center', va='center', fontsize=10, color='red')

    # Row 2: Blobs dataset
    # Original data
    axes[1, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=np.zeros(len(X_blobs)), 
                       s=20, cmap='tab10', edgecolors='none')
    axes[1, 0].set_title("Blobs Data (Original)")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("x2")

    # K-means
    axes[1, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_blobs_kmeans, 
                       s=20, cmap='tab10', edgecolors='none')
    axes[1, 1].set_title("K-means on Blobs (k=3)")
    axes[1, 1].set_xlabel("x1")
    axes[1, 1].set_ylabel("x2")

    # DBSCAN
    plot_clustering_results(X_blobs, labels_blobs_dbscan, 
                           "DBSCAN on Blobs (eps=1.5, min_pts=5)", axes[1, 2])

    # Add empty plot for spacing or additional info
    axes[1, 3].axis('off')
    axes[1, 3].text(0.5, 0.7, "Blobs Dataset:", 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    axes[1, 3].text(0.5, 0.5, "✓ Both work well\non spherical clusters", 
                    ha='center', va='center', fontsize=10, color='green')
    axes[1, 3].text(0.5, 0.3, "✓ DBSCAN can\nidentify noise", 
                    ha='center', va='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

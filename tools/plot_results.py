import argparse
import csv

import numpy as np


def read_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols: dict[str, list[float]] = {}
        for row in r:
            for k, v in row.items():
                if v is None or v == "":
                    continue
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def inertia(X: np.ndarray, labels: np.ndarray) -> float:
    k = int(labels.max()) + 1
    centroids = np.zeros((k, X.shape[1]), dtype=float)
    counts = np.zeros((k,), dtype=int)
    for i, c in enumerate(labels.astype(int)):
        centroids[c] += X[i]
        counts[c] += 1
    for c in range(k):
        if counts[c] > 0:
            centroids[c] /= counts[c]
    return float(np.sum((X - centroids[labels.astype(int)]) ** 2))


def try_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception:
        return None
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--pred", default="pred.csv")
    p.add_argument("--show", action="store_true")
    p.add_argument("--out", default="clusters.png")
    args = p.parse_args()

    data = read_csv(args.data)
    pred = read_csv(args.pred)

    X = np.stack([data["x1"], data["x2"]], axis=1)
    pred_labels = pred["pred_label"].astype(int)

    inert = inertia(X, pred_labels)
    sil = try_silhouette(X, pred_labels)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"Metrics: inertia={inert:.6g}, silhouette={(sil if sil is not None else 'n/a')}")
        raise SystemExit(f"matplotlib not available: {e}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    if "true_label" in data:
        y_true = data["true_label"].astype(int)
        ax[0].scatter(X[:, 0], X[:, 1], c=y_true, s=18, cmap="tab10")
        ax[0].set_title("True labels")
    else:
        ax[0].scatter(X[:, 0], X[:, 1], s=18)
        ax[0].set_title("Data")

    ax[1].scatter(X[:, 0], X[:, 1], c=pred_labels, s=18, cmap="tab10", vmin=-1, vmax=pred_labels.max())
    
    # Определяем тип алгоритма по наличию шума (-1 метки)
    has_noise = (pred_labels == -1).any()
    algorithm_name = "DBSCAN" if has_noise else "K-means"
    
    title = f"{algorithm_name} pred\ninertia={inert:.4g}"
    if sil is not None:
        title += f", silhouette={sil:.4g}"
    ax[1].set_title(title)

    for a in ax:
        a.set_xlabel("x1")
        a.set_ylabel("x2")

    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")
    print(f"Metrics: inertia={inert:.6g}, silhouette={(sil if sil is not None else 'n/a')}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


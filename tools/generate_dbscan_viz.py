#!/usr/bin/env python3
"""
Генерация примера визуализации DBSCAN на синтетических данных.
Создаёт датасет с полумесяцами (moons) и применяет DBSCAN для кластеризации.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans


def plot_comparison(X_moons, X_blobs, out_path: str = "assets/dbscan_visualization.png"):
    """
    Создаёт сравнительную визуализацию K-means и DBSCAN на разных типах данных.
    
    Args:
        X_moons: Данные moon (полумесяцы)
        X_blobs: Данные blobs (сферические кластеры)
        out_path: Путь для сохранения изображения
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # === Верхний ряд: Moons (полумесяцы) ===
    axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c='gray', s=30, alpha=0.7)
    axes[0, 0].set_title("Исходные данные (Moons)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("X1")
    axes[0, 0].set_ylabel("X2")
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # K-means на moons
    kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_km_moons = kmeans_moons.fit_predict(X_moons)
    
    scatter_km_moons = axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], 
                                           c=labels_km_moons, cmap='tab10', s=30, alpha=0.7)
    axes[0, 1].set_title("K-means (k=2)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("X1")
    axes[0, 1].set_ylabel("X2")
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Добавляем центроиды
    centers_km = kmeans_moons.cluster_centers_
    axes[0, 1].scatter(centers_km[:, 0], centers_km[:, 1], 
                       c='red', s=200, marker='X', edgecolors='black', linewidths=2, label='Centroids')
    axes[0, 1].legend(loc='upper right')
    
    # DBSCAN на moons
    dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
    labels_db_moons = dbscan_moons.fit_predict(X_moons)
    
    scatter_db_moons = axes[0, 2].scatter(X_moons[:, 0], X_moons[:, 1], 
                                           c=labels_db_moons, cmap='tab10', s=30, alpha=0.7)
    axes[0, 2].set_title(f"DBSCAN (eps=0.2, min_samples=5)\nКластеров: {len(set(labels_db_moons)) - (1 if -1 in labels_db_moons else 0)}", 
                         fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel("X1")
    axes[0, 2].set_ylabel("X2")
    axes[0, 2].set_aspect('equal')
    axes[0, 2].grid(True, alpha=0.3)
    
    # === Нижний ряд: Blobs (сферические кластеры) ===
    axes[1, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c='gray', s=30, alpha=0.7)
    axes[1, 0].set_title("Исходные данные (Blobs)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("X1")
    axes[1, 0].set_ylabel("X2")
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # K-means на blobs
    kmeans_blobs = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km_blobs = kmeans_blobs.fit_predict(X_blobs)
    
    scatter_km_blobs = axes[1, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                                           c=labels_km_blobs, cmap='tab10', s=30, alpha=0.7)
    axes[1, 1].set_title("K-means (k=3)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("X1")
    axes[1, 1].set_ylabel("X2")
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Добавляем центроиды
    centers_km_blobs = kmeans_blobs.cluster_centers_
    axes[1, 1].scatter(centers_km_blobs[:, 0], centers_km_blobs[:, 1], 
                       c='red', s=200, marker='X', edgecolors='black', linewidths=2, label='Centroids')
    axes[1, 1].legend(loc='upper right')
    
    # DBSCAN на blobs
    dbscan_blobs = DBSCAN(eps=0.5, min_samples=5)
    labels_db_blobs = dbscan_blobs.fit_predict(X_blobs)
    
    scatter_db_blobs = axes[1, 2].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                                           c=labels_db_blobs, cmap='tab10', s=30, alpha=0.7)
    n_clusters_blobs = len(set(labels_db_blobs)) - (1 if -1 in labels_db_blobs else 0)
    n_noise_blobs = list(labels_db_blobs).count(-1)
    title_db_blobs = f"DBSCAN (eps=0.5, min_samples=5)\nКластеров: {n_clusters_blobs}, Шум: {n_noise_blobs}"
    axes[1, 2].set_title(title_db_blobs, fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel("X1")
    axes[1, 2].set_ylabel("X2")
    axes[1, 2].set_aspect('equal')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Визуализация сохранена в: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Генерация примера визуализации DBSCAN")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--samples", type=int, default=300, help="Количество образцов")
    parser.add_argument("--noise", type=float, default=0.1, help="Уровень шума для moons")
    parser.add_argument("--out", type=str, default="assets/dbscan_visualization.png", 
                        help="Путь для сохранения изображения")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Генерация данных moon (полумесяцы)
    X_moons, _ = make_moons(n_samples=args.samples, noise=args.noise, random_state=args.seed)
    
    # Генерация данных blobs (сферические кластеры)
    X_blobs, _ = make_blobs(n_samples=args.samples, centers=3, cluster_std=1.0, 
                            random_state=args.seed)
    
    plot_comparison(X_moons, X_blobs, out_path=args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Генерация отдельной визуализации только для DBSCAN.
Создаёт датасет с полумесяцами (moons) и применяет только DBSCAN.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


def plot_dbscan_only(out_path: str = "assets/dbscan_only.png"):
    """
    Создаёт визуализацию работы только алгоритма DBSCAN.
    
    Args:
        out_path: Путь для сохранения изображения
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Генерация данных moon (полумесяцы)
    np.random.seed(42)
    X, _ = make_moons(n_samples=400, noise=0.1, random_state=42)
    
    # === Левая панель: Исходные данные ===
    axes[0].scatter(X[:, 0], X[:, 1], c='steelblue', s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
    axes[0].set_title("Исходные данные", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("X1", fontsize=12)
    axes[0].set_ylabel("X2", fontsize=12)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(-1.5, 2.5)
    axes[0].set_ylim(-1.0, 1.0)
    
    # === Правая панель: Результат DBSCAN ===
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, 
                               alpha=0.7, edgecolors='white', linewidths=0.5)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    title_text = f"Результат DBSCAN\nКластеров: {n_clusters}, Шум: {n_noise}"
    axes[1].set_title(title_text, fontsize=14, fontweight='bold')
    axes[1].set_xlabel("X1", fontsize=12)
    axes[1].set_ylabel("X2", fontsize=12)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(-1.5, 2.5)
    axes[1].set_ylim(-1.0, 1.0)
    
    # Добавляем легенду
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gray', label='Шум'),
                       Patch(facecolor=plt.cm.tab10(0), label='Кластер 0'),
                       Patch(facecolor=plt.cm.tab10(1), label='Кластер 1')]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Визуализация DBSCAN сохранена в: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Генерация отдельной визуализации DBSCAN")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--samples", type=int, default=400, help="Количество образцов")
    parser.add_argument("--noise", type=float, default=0.1, help="Уровень шума")
    parser.add_argument("--eps", type=float, default=0.2, help="Параметр eps для DBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="Параметр min_samples для DBSCAN")
    parser.add_argument("--out", type=str, default="assets/dbscan_only.png", 
                        help="Путь для сохранения изображения")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Генерация данных moon (полумесяцы)
    X, _ = make_moons(n_samples=args.samples, noise=args.noise, random_state=args.seed)
    
    # Применяем DBSCAN
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN нашёл {n_clusters} кластеров и {n_noise} точек шума")
    
    # Создаём визуализацию
    plot_dbscan_only(out_path=args.out)


if __name__ == "__main__":
    main()

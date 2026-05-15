/**
 * @file dbscan.c
 * @brief Реализация алгоритма кластеризации DBSCAN.
 * 
 * Алгоритм DBSCAN группирует точки на основе их плотности распределения.
 * Основные понятия:
 * - Core point (ядерная точка): точка, у которой в радиусе eps находится не менее min_pts соседей.
 * - Border point (граничная точка): точка, которая находится в окрестности ядерной, но сама не является ядерной.
 * - Noise point (шум): точка, не относящаяся ни к одному кластеру.
 */

#include "dbscan.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief Вычисляет евклидово расстояние между двумя точками.
 * 
 * @param a Указатель на координаты первой точки.
 * @param b Указатель на координаты второй точки.
 * @param dim Размерность пространства.
 * @return Евклидово расстояние между точками.
 */
static double euclidean_distance(const double *a, const double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/**
 * @brief Находит всех соседей точки в радиусе eps.
 * 
 * @param dataset Датасет с данными.
 * @param point_idx Индекс текущей точки.
 * @param eps Радиус поиска соседей.
 * @param neighbors Массив для хранения индексов соседей (должен быть достаточно большим).
 * @return Количество найденных соседей.
 */
static int find_neighbors(Dataset *dataset, int point_idx, double eps, int *neighbors) {
    int count = 0;
    double *point = dataset->data[point_idx];
    
    for (int i = 0; i < dataset->num_rows; i++) {
        if (i == point_idx) continue;
        
        double dist = euclidean_distance(point, dataset->data[i], dataset->num_cols);
        if (dist <= eps) {
            neighbors[count++] = i;
        }
    }
    
    return count;
}

/**
 * @brief Расширяет кластер из данной точки.
 * 
 * @param dataset Датасет с данными.
 * @param point_idx Индекс начальной точки.
 * @param eps Радиус eps.
 * @param min_pts Минимальное количество точек min_pts.
 * @param labels Массив меток кластеров.
 * @param cluster_id Идентификатор текущего кластера.
 * @param visited Массив посещенных точек.
 */
static void expand_cluster(Dataset *dataset, int point_idx, double eps, int min_pts,
                          int *labels, int cluster_id, int *visited) {
    int *neighbors = (int *)malloc(dataset->num_rows * sizeof(int));
    if (!neighbors) {
        fprintf(stderr, "Ошибка выделения памяти для соседей\n");
        return;
    }
    
    int neighbor_count = find_neighbors(dataset, point_idx, eps, neighbors);
    
    // Если точка является ядерной (имеет достаточно соседей)
    if (neighbor_count >= min_pts) {
        labels[point_idx] = cluster_id;
        
        // Обрабатываем всех соседей
        for (int i = 0; i < neighbor_count; i++) {
            int neighbor_idx = neighbors[i];
            
            if (visited[neighbor_idx] == 0) {
                visited[neighbor_idx] = 1;
                
                // Рекурсивно расширяем кластер для соседей
                int *sub_neighbors = (int *)malloc(dataset->num_rows * sizeof(int));
                if (sub_neighbors) {
                    int sub_count = find_neighbors(dataset, neighbor_idx, eps, sub_neighbors);
                    
                    if (sub_count >= min_pts) {
                        // Если сосед тоже ядерная точка, добавляем его соседей
                        for (int j = 0; j < sub_count; j++) {
                            int found = 0;
                            for (int k = 0; k < neighbor_count; k++) {
                                if (neighbors[k] == sub_neighbors[j]) {
                                    found = 1;
                                    break;
                                }
                            }
                            if (!found) {
                                neighbors[neighbor_count++] = sub_neighbors[j];
                            }
                        }
                    }
                    free(sub_neighbors);
                }
                
                // Если точка еще не принадлежит кластеру, добавляем её
                if (labels[neighbor_idx] == -1) {
                    labels[neighbor_idx] = cluster_id;
                }
            } else if (labels[neighbor_idx] == -1) {
                // Точка уже посещена, но не принадлежит кластеру - добавляем как граничную
                labels[neighbor_idx] = cluster_id;
            }
        }
    }
    
    free(neighbors);
}

DbscanResult dbscan_cluster(Dataset *dataset, double eps, int min_pts) {
    DbscanResult result;
    result.num_points = dataset->num_rows;
    result.num_clusters = 0;
    
    // Выделяем память под метки кластеров
    result.labels = (int *)malloc(dataset->num_rows * sizeof(int));
    if (!result.labels) {
        fprintf(stderr, "Ошибка выделения памяти для меток\n");
        result.num_points = 0;
        return result;
    }
    
    // Инициализируем все точки как шум (-1)
    for (int i = 0; i < dataset->num_rows; i++) {
        result.labels[i] = -1;
    }
    
    // Массив посещенных точек
    int *visited = (int *)calloc(dataset->num_rows, sizeof(int));
    if (!visited) {
        fprintf(stderr, "Ошибка выделения памяти для массива посещений\n");
        free(result.labels);
        result.labels = NULL;
        result.num_points = 0;
        return result;
    }
    
    // Основной цикл DBSCAN
    for (int i = 0; i < dataset->num_rows; i++) {
        if (visited[i] == 0) {
            visited[i] = 1;
            
            int *neighbors = (int *)malloc(dataset->num_rows * sizeof(int));
            if (!neighbors) {
                fprintf(stderr, "Ошибка выделения памяти для соседей\n");
                continue;
            }
            
            int neighbor_count = find_neighbors(dataset, i, eps, neighbors);
            
            // Если точка является ядерной, создаем новый кластер
            if (neighbor_count >= min_pts) {
                result.num_clusters++;
                expand_cluster(dataset, i, eps, min_pts, result.labels, result.num_clusters, visited);
            }
            
            free(neighbors);
        }
    }
    
    free(visited);
    return result;
}

void dbscan_free_result(DbscanResult *result) {
    if (result && result->labels) {
        free(result->labels);
        result->labels = NULL;
    }
}

/**
 * @file dbscan.h
 * @brief Заголовочный файл реализации алгоритма кластеризации DBSCAN.
 * 
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) — 
 * алгоритм кластеризации на основе плотности. Он группирует точки, 
 * находящиеся близко друг к другу, и помечает изолированные точки как шум.
 */

#ifndef DBSCAN_H
#define DBSCAN_H

#include <stddef.h>

/**
 * @brief Структура данных для передачи датасета в алгоритмы кластеризации.
 */
typedef struct {
    double **data;      /**< Массив указателей на строки данных [num_rows][num_cols] */
    int num_rows;       /**< Количество строк (точек данных) */
    int num_cols;       /**< Количество столбцов (признаков) */
} Dataset;

/**
 * @brief Структура результатов кластеризации DBSCAN.
 */
typedef struct {
    int *labels;          /**< Метка кластера для каждой точки (-1 означает шум). */
    int num_clusters;     /**< Количество найденных кластеров. */
    int num_points;       /**< Общее количество точек данных. */
} DbscanResult;

/**
 * @brief Выполняет кластеризацию DBSCAN.
 * 
 * @param dataset Указатель на структуру датасета с данными.
 * @param eps Радиус окрестности для поиска соседей (параметр Eps).
 * @param min_pts Минимальное количество точек для образования плотной области (параметр MinPts).
 * @return Структура DbscanResult с метками кластеров и статистикой.
 */
DbscanResult dbscan_cluster(Dataset *dataset, double eps, int min_pts);

/**
 * @brief Освобождает память, выделенную под результаты DBSCAN.
 * 
 * @param result Указатель на структуру результатов DbscanResult.
 */
void dbscan_free_result(DbscanResult *result);

#endif /* DBSCAN_H */

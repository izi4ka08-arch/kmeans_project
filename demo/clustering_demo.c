/**
 * @file clustering_demo.c
 * @brief Демо-приложение для кластеризации с выбором алгоритма (K-means или DBSCAN).
 * 
 * Приложение позволяет выбрать алгоритм кластеризации через командную строку:
 * - kmeans: классический алгоритм K-means
 * - dbscan: алгоритм кластеризации на основе плотности DBSCAN
 * 
 * Примеры использования:
 *   ./clustering_demo data.csv 3 kmeans           # K-means с 3 кластерами
 *   ./clustering_demo data.csv 0.5 4 dbscan       # DBSCAN с eps=0.5, min_pts=4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/csv.h"
#include "../src/kmeans.h"
#include "../src/dbscan.h"

/**
 * @brief Структура конфигурации для разных алгоритмов.
 */
typedef struct {
    char algorithm[32];     /**< Название алгоритма: "kmeans" или "dbscan" */
    
    // Параметры для K-means
    int k;                  /**< Количество кластеров для K-means */
    int max_iter;           /**< Максимальное количество итераций */
    int init_method;        /**< Метод инициализации центроидов */
    
    // Параметры для DBSCAN
    double eps;             /**< Радиус окрестности для DBSCAN */
    int min_pts;            /**< Минимальное количество точек для DBSCAN */
} ClusteringConfig;

/**
 * @brief Выводит справку по использованию программы.
 * 
 * @param program_name Имя программы (argv[0]).
 */
void print_usage(const char *program_name) {
    printf("Использование:\n");
    printf("  %s <input.csv> <params...> <algorithm>\n\n", program_name);
    
    printf("Алгоритмы:\n");
    printf("  kmeans  - Кластеризация K-means\n");
    printf("  dbscan  - Кластеризация DBSCAN\n\n");
    
    printf("Параметры для K-means:\n");
    printf("  %s <input.csv> <k> [max_iter] [init_method] kmeans\n", program_name);
    printf("    k          - количество кластеров (целое число > 0)\n");
    printf("    max_iter   - макс. итераций (по умолчанию 100)\n");
    printf("    init_method- метод инициализации: 0=random, 1=kmeans++ (по умолчанию 1)\n\n");
    
    printf("Параметры для DBSCAN:\n");
    printf("  %s <input.csv> <eps> <min_pts> dbscan\n", program_name);
    printf("    eps        - радиус окрестности (вещественное число > 0)\n");
    printf("    min_pts    - мин. количество точек (целое число > 0)\n\n");
    
    printf("Примеры:\n");
    printf("  %s data.csv 3 kmeans\n", program_name);
    printf("  %s data.csv 3 200 0 kmeans\n", program_name);
    printf("  %s data.csv 0.5 4 dbscan\n", program_name);
}

/**
 * @brief Выполняет кластеризацию K-means.
 * 
 * @param dataset Датасет для кластеризации.
 * @param config Конфигурация с параметрами K-means.
 * @return Указатель на массив меток кластеров.
 */
int* run_kmeans(Dataset *dataset, ClusteringConfig *config) {
    printf("\n=== Запуск K-means ===\n");
    printf("Количество кластеров (k): %d\n", config->k);
    printf("Максимум итераций: %d\n", config->max_iter);
    printf("Метод инициализации: %s\n", 
           config->init_method == 0 ? "random" : "kmeans++");
    
    // Подготовка параметров
    kmeans_params_t params = {0};
    params.k = config->k;
    params.max_iters = config->max_iter;
    params.tol = 1e-4;
    params.seed = 42;
    params.init_method = (config->init_method == 0) ? 
                         KMEANS_INIT_RANDOM : KMEANS_INIT_KMEANS_PLUS_PLUS;
    
    // Выделение памяти для центроидов и меток
    double *centroids = (double *)malloc(config->k * dataset->num_cols * sizeof(double));
    int *labels = (int *)malloc(dataset->num_rows * sizeof(int));
    double inertia = 0.0;
    
    if (!centroids || !labels) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        free(centroids);
        free(labels);
        return NULL;
    }
    
    // Преобразование данных в плоский массив для kmeans_fit
    double *X_flat = (double *)malloc(dataset->num_rows * dataset->num_cols * sizeof(double));
    if (!X_flat) {
        fprintf(stderr, "Ошибка выделения памяти для данных\n");
        free(centroids);
        free(labels);
        return NULL;
    }
    
    for (int i = 0; i < dataset->num_rows; i++) {
        for (int j = 0; j < dataset->num_cols; j++) {
            X_flat[i * dataset->num_cols + j] = dataset->data[i][j];
        }
    }
    
    // Запуск K-means
    int rc = kmeans_fit(X_flat, dataset->num_rows, dataset->num_cols, 
                        &params, labels, centroids, &inertia);
    
    free(X_flat);
    free(centroids);
    
    if (rc != 0) {
        fprintf(stderr, "Ошибка выполнения K-means: %d\n", rc);
        free(labels);
        return NULL;
    }
    
    printf("Финальное значение инерции: %.6f\n", inertia);
    printf("Найдено кластеров: %d\n", config->k);
    
    return labels;
}

/**
 * @brief Выполняет кластеризацию DBSCAN.
 * 
 * @param dataset Датасет для кластеризации.
 * @param config Конфигурация с параметрами DBSCAN.
 * @return Указатель на массив меток кластеров.
 */
int* run_dbscan(Dataset *dataset, ClusteringConfig *config) {
    printf("\n=== DBSCAN Task ===\n");
    printf("Radius eps: %.4f\n", config->eps);
    printf("Minimum points (min_pts): %d\n", config->min_pts);
    
    DbscanResult result = dbscan_cluster(
        dataset,
        config->eps,
        config->min_pts
    );
    
    printf("Clusters found: %d\n", result.num_clusters);
    
    // Подсчитываем количество точек шума
    int noise_count = 0;
    for (int i = 0; i < result.num_points; i++) {
        if (result.labels[i] == -1) {
            noise_count++;
        }
    }
    printf("Points marked as noise: %d (%.2f%%)\n", 
           noise_count, 
           100.0 * noise_count / result.num_points);
    
    // Копируем результаты в новый массив для возврата
    int *labels = (int *)malloc(result.num_points * sizeof(int));
    if (labels) {
        memcpy(labels, result.labels, result.num_points * sizeof(int));
    }
    
    dbscan_free_result(&result);
    return labels;
}

/**
 * @brief Сохраняет результаты кластеризации в CSV файл.
 * 
 * @param input_path Путь к входному файлу.
 * @param ds Исходный датасет.
 * @param labels Метки кластеров.
 * @param num_points Количество точек.
 */
void save_results(const char *input_path, csv_dataset_t *ds, int *labels, int num_points) {
    const char *output_path = "pred.csv";
    
    if (csv_write_predictions(output_path, ds, labels) == 0) {
        printf("successfull\n");
    } else {
        fprintf(stderr, "Ошибка записи результатов в файл: %s\n", output_path);
    }
}

int main(int argc, char *argv[]) {
    // Проверка количества аргументов
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    ClusteringConfig config = {0};
    config.max_iter = 100;      // Значение по умолчанию
    config.init_method = 1;     // kmeans++ по умолчанию
    
    // Парсинг аргументов в зависимости от алгоритма
    const char *input_file = argv[1];
    const char *algorithm = argv[argc - 1];
    
    strncpy(config.algorithm, algorithm, sizeof(config.algorithm) - 1);
    
    if (strcmp(algorithm, "kmeans") == 0) {
        // Парсинг параметров для K-means
        config.k = atoi(argv[2]);
        if (config.k <= 0) {
            fprintf(stderr, "Ошибка: k должно быть положительным числом\n");
            return 1;
        }
        
        if (argc >= 5) {
            config.max_iter = atoi(argv[3]);
        }
        if (argc >= 6) {
            config.init_method = atoi(argv[4]);
        }
        
    } else if (strcmp(algorithm, "dbscan") == 0) {
        // Парсинг параметров для DBSCAN
        if (argc != 5) {
            fprintf(stderr, "Ошибка: DBSCAN требует ровно 2 параметра (eps и min_pts)\n");
            print_usage(argv[0]);
            return 1;
        }
        
        config.eps = atof(argv[2]);
        config.min_pts = atoi(argv[3]);
        
        if (config.eps <= 0) {
            fprintf(stderr, "Ошибка: eps должно быть положительным числом\n");
            return 1;
        }
        if (config.min_pts <= 0) {
            fprintf(stderr, "Ошибка: min_pts должно быть положительным числом\n");
            return 1;
        }
        
    } else {
        fprintf(stderr, "Ошибка: неизвестный алгоритм '%s'\n", algorithm);
        print_usage(argv[0]);
        return 1;
    }
    
    // Чтение датасета из CSV
    printf("Чтение датасета из файла: %s\n", input_file);
    csv_dataset_t ds = {0};
    
    // Определяем количество признаков (предполагаем, что все колонки - признаки)
    int n_features = 2;  // По умолчанию для демонстрации
    if (csv_read_dataset(input_file, n_features, -1, &ds) != 0) {
        fprintf(stderr, "Ошибка чтения файла: %s\n", input_file);
        return 1;
    }
    
    printf("Загружено %d образцов с %d признаками\n", ds.n_samples, ds.n_features);
    
    // Преобразуем csv_dataset_t в Dataset для библиотеки
    Dataset dataset = {0};
    dataset.data = (double **)malloc(ds.n_samples * sizeof(double *));
    if (!dataset.data) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        csv_free_dataset(&ds);
        return 1;
    }
    
    dataset.num_rows = ds.n_samples;
    dataset.num_cols = ds.n_features;
    
    for (int i = 0; i < ds.n_samples; i++) {
        dataset.data[i] = ds.X + i * ds.n_features;
    }
    
    // Запуск выбранного алгоритма
    int *labels = NULL;
    
    if (strcmp(algorithm, "kmeans") == 0) {
        labels = run_kmeans(&dataset, &config);
    } else if (strcmp(algorithm, "dbscan") == 0) {
        labels = run_dbscan(&dataset, &config);
    }
    
    if (!labels) {
        fprintf(stderr, "Ошибка выполнения кластеризации\n");
        free(dataset.data);
        csv_free_dataset(&ds);
        return 1;
    }
    
    // Сохранение результатов
    save_results(input_file, &ds, labels, ds.n_samples);
    
    // Очистка памяти
    free(labels);
    free(dataset.data);
    csv_free_dataset(&ds);
    
    printf("\nClustering completed successfully!\n");
    return 0;
}

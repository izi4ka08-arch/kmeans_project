/**
 * @file clustering_demo.c
 * @brief Demo application for clustering with algorithm selection (K-means or DBSCAN).
 * 
 * Application allows selecting clustering algorithm via command line:
 * - kmeans: classic K-means algorithm
 * - dbscan: density-based clustering algorithm DBSCAN
 * 
 * Usage examples:
 *   ./clustering_demo data.csv 3 kmeans           # K-means with 3 clusters
 *   ./clustering_demo data.csv 0.5 4 dbscan       # DBSCAN with eps=0.5, min_pts=4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/csv.h"
#include "../src/kmeans.h"
#include "../src/dbscan.h"

/**
 * @brief Configuration structure for different algorithms.
 */
typedef struct {
    char algorithm[32];     /**< Algorithm name: "kmeans" or "dbscan" */
    
    // Parameters for K-means
    int k;                  /**< Number of clusters for K-means */
    int max_iter;           /**< Maximum number of iterations */
    int init_method;        /**< Centroid initialization method */
    
    // Parameters for DBSCAN
    double eps;             /**< Neighborhood radius for DBSCAN */
    int min_pts;            /**< Minimum number of points for DBSCAN */
} ClusteringConfig;

/**
 * @brief Prints usage information.
 * 
 * @param program_name Program name (argv[0]).
 */
void print_usage(const char *program_name) {
    printf("Usage:\n");
    printf("  %s <input.csv> <params...> <algorithm>\n\n", program_name);
    
    printf("Algorithms:\n");
    printf("  kmeans  - K-means clustering\n");
    printf("  dbscan  - DBSCAN clustering\n\n");
    
    printf("Parameters for K-means:\n");
    printf("  %s <input.csv> <k> [max_iter] [init_method] kmeans\n", program_name);
    printf("    k          - number of clusters (integer > 0)\n");
    printf("    max_iter   - maximum iterations (default: 100)\n");
    printf("    init_method- initialization method: 0=random, 1=kmeans++ (default: 1)\n\n");
    
    printf("Parameters for DBSCAN:\n");
    printf("  %s <input.csv> <eps> <min_pts> dbscan\n", program_name);
    printf("    eps        - neighborhood radius (float > 0)\n");
    printf("    min_pts    - minimum number of points (integer > 0)\n\n");
    
    printf("Examples:\n");
    printf("  %s data.csv 3 kmeans\n", program_name);
    printf("  %s data.csv 3 200 0 kmeans\n", program_name);
    printf("  %s data.csv 0.5 4 dbscan\n", program_name);
}

/**
 * @brief Performs K-means clustering.
 * 
 * @param dataset Dataset for clustering.
 * @param config Configuration with K-means parameters.
 * @return Pointer to array of cluster labels.
 */
int* run_kmeans(Dataset *dataset, ClusteringConfig *config) {
    printf("\n=== K-means Task ===\n");
    printf("Number of clusters (k): %d\n", config->k);
    printf("Maximum iterations: %d\n", config->max_iter);
    printf("Initialization method: %s\n", 
           config->init_method == 0 ? "random" : "kmeans++");
    
    // Prepare parameters
    kmeans_params_t params = {0};
    params.k = config->k;
    params.max_iters = config->max_iter;
    params.tol = 1e-4;
    params.seed = 42;
    params.init_method = (config->init_method == 0) ? 
                         KMEANS_INIT_RANDOM : KMEANS_INIT_KMEANS_PLUS_PLUS;
    
    // Allocate memory for centroids and labels
    double *centroids = (double *)malloc(config->k * dataset->num_cols * sizeof(double));
    int *labels = (int *)malloc(dataset->num_rows * sizeof(int));
    double inertia = 0.0;
    
    if (!centroids || !labels) {
        fprintf(stderr, "Memory allocation error\n");
        free(centroids);
        free(labels);
        return NULL;
    }
    
    // Convert data to flat array for kmeans_fit
    double *X_flat = (double *)malloc(dataset->num_rows * dataset->num_cols * sizeof(double));
    if (!X_flat) {
        fprintf(stderr, "Memory allocation error for data\n");
        free(centroids);
        free(labels);
        return NULL;
    }
    
    for (int i = 0; i < dataset->num_rows; i++) {
        for (int j = 0; j < dataset->num_cols; j++) {
            X_flat[i * dataset->num_cols + j] = dataset->data[i][j];
        }
    }
    
    // Run K-means
    int rc = kmeans_fit(X_flat, dataset->num_rows, dataset->num_cols, 
                        &params, labels, centroids, &inertia);
    
    free(X_flat);
    free(centroids);
    
    if (rc != 0) {
        fprintf(stderr, "K-means execution error: %d\n", rc);
        free(labels);
        return NULL;
    }
    
    printf("Final inertia value: %.6f\n", inertia);
    printf("Number of clusters found: %d\n", config->k);
    
    return labels;
}

/**
 * @brief Performs DBSCAN clustering.
 * 
 * @param dataset Dataset for clustering.
 * @param config Configuration with DBSCAN parameters.
 * @return Pointer to array of cluster labels.
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
        // Parse parameters for K-means
        config.k = atoi(argv[2]);
        if (config.k <= 0) {
            fprintf(stderr, "Error: k must be a positive integer\n");
            return 1;
        }
        
        if (argc >= 5) {
            config.max_iter = atoi(argv[3]);
        }
        if (argc >= 6) {
            config.init_method = atoi(argv[4]);
        }
        
    } else if (strcmp(algorithm, "dbscan") == 0) {
        // Parse parameters for DBSCAN
        if (argc != 5) {
            fprintf(stderr, "Error: DBSCAN requires exactly 2 parameters (eps and min_pts)\n");
            print_usage(argv[0]);
            return 1;
        }
        
        config.eps = atof(argv[2]);
        config.min_pts = atoi(argv[3]);
        
        if (config.eps <= 0) {
            fprintf(stderr, "Error: eps must be a positive number\n");
            return 1;
        }
        if (config.min_pts <= 0) {
            fprintf(stderr, "Error: min_pts must be a positive integer\n");
            return 1;
        }
        
    } else {
        fprintf(stderr, "Error: unknown algorithm '%s'\n", algorithm);
        print_usage(argv[0]);
        return 1;
    }
    
    // Read dataset from CSV
    printf("Reading dataset from file: %s\n", input_file);
    csv_dataset_t ds = {0};
    
    // Determine number of features (assume all columns are features)
    int n_features = 2;  // Default for demonstration
    if (csv_read_dataset(input_file, n_features, -1, &ds) != 0) {
        fprintf(stderr, "Error reading file: %s\n", input_file);
        return 1;
    }
    
    printf("Loaded %d samples with %d features\n", ds.n_samples, ds.n_features);
    
    // Convert csv_dataset_t to Dataset for the library
    Dataset dataset = {0};
    dataset.data = (double **)malloc(ds.n_samples * sizeof(double *));
    if (!dataset.data) {
        fprintf(stderr, "Error allocating memory\n");
        csv_free_dataset(&ds);
        return 1;
    }
    
    dataset.num_rows = ds.n_samples;
    dataset.num_cols = ds.n_features;
    
    for (int i = 0; i < ds.n_samples; i++) {
        dataset.data[i] = ds.X + i * ds.n_features;
    }
    
    // Run selected algorithm
    int *labels = NULL;
    
    if (strcmp(algorithm, "kmeans") == 0) {
        labels = run_kmeans(&dataset, &config);
    } else if (strcmp(algorithm, "dbscan") == 0) {
        labels = run_dbscan(&dataset, &config);
    }
    
    if (!labels) {
        fprintf(stderr, "Clustering execution error\n");
        free(dataset.data);
        csv_free_dataset(&ds);
        return 1;
    }
    
    // Save results
    save_results(input_file, &ds, labels, ds.n_samples);
    
    // Free memory
    free(labels);
    free(dataset.data);
    csv_free_dataset(&ds);
    
    printf("\nClustering completed successfully!\n");
    return 0;
}

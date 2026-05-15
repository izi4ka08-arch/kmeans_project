/**
 * @file kmeans_demo.c
 * @brief Демонстрационное приложение для K-means кластеризации
 * 
 * Эта программа читает данные из CSV, выполняет K-means кластеризацию
 * и записывает результаты с предсказанными метками в выходной файл.
 */

#include "csv.h"
#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Выводит справку по использованию программы
 * 
 * @param prog Имя программы (argv[0])
 */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --in data.csv --k 3 --out pred.csv [--features 2] [--label-col 2]\n"
        "     [--max-iters 100] [--tol 1e-4] [--seed 42] [--init random|kpp] [--no-true]\n",
        prog
    );
}

/**
 * @brief Сравнивает две строки на равенство
 * 
 * @param a Первая строка
 * @param b Вторая строка
 * @return 1 если строки равны, 0 иначе
 */
static int str_eq(const char* a, const char* b) { return a && b && strcmp(a, b) == 0; }

int main(int argc, char** argv) {
    /* Параметры по умолчанию */
    const char* in_path = NULL;
    const char* out_path = "pred.csv";
    int k = 0;
    int n_features = 2;
    int label_col = 2;
    int max_iters = 100;
    double tol = 1e-4;
    unsigned int seed = 42;
    kmeans_init_method_t init_method = KMEANS_INIT_KMEANS_PLUS_PLUS;

    /* Парсинг аргументов командной строки */
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (str_eq(a, "--in") && i + 1 < argc) in_path = argv[++i];
        else if (str_eq(a, "--out") && i + 1 < argc) out_path = argv[++i];
        else if (str_eq(a, "--k") && i + 1 < argc) k = atoi(argv[++i]);
        else if (str_eq(a, "--features") && i + 1 < argc) n_features = atoi(argv[++i]);
        else if (str_eq(a, "--label-col") && i + 1 < argc) label_col = atoi(argv[++i]);
        else if (str_eq(a, "--max-iters") && i + 1 < argc) max_iters = atoi(argv[++i]);
        else if (str_eq(a, "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (str_eq(a, "--seed") && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (str_eq(a, "--init") && i + 1 < argc) {
            const char* v = argv[++i];
            if (str_eq(v, "random")) init_method = KMEANS_INIT_RANDOM;
            else if (str_eq(v, "kpp")) init_method = KMEANS_INIT_KMEANS_PLUS_PLUS;
            else { usage(argv[0]); return 2; }
        } else if (str_eq(a, "--no-true")) {
            label_col = -1;  /* Игнорировать столбец с истинными метками */
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    /* Проверка обязательных параметров */
    if (!in_path || k <= 0) { usage(argv[0]); return 2; }

    /* Чтение датасета из CSV */
    csv_dataset_t ds;
    int rc = csv_read_dataset(in_path, n_features, label_col, &ds);
    if (rc != 0) { fprintf(stderr, "csv_read_dataset failed: %d\n", rc); return 3; }

    /* Выделение памяти для результатов */
    int* labels = (int*)malloc((size_t)ds.n_samples * sizeof(int));
    double* centroids = (double*)malloc((size_t)k * (size_t)ds.n_features * sizeof(double));
    if (!labels || !centroids) {
        fprintf(stderr, "out of memory\n");
        csv_free_dataset(&ds);
        free(labels);
        free(centroids);
        return 4;
    }

    /* Настройка параметров K-means */
    kmeans_params_t params;
    params.k = k;
    params.max_iters = max_iters;
    params.tol = tol;
    params.seed = seed;
    params.init_method = init_method;

    /* Запуск K-means */
    double inertia = 0.0;
    rc = kmeans_fit(ds.X, ds.n_samples, ds.n_features, &params, labels, centroids, &inertia);
    if (rc != 0) {
        fprintf(stderr, "kmeans_fit failed: %d\n", rc);
        csv_free_dataset(&ds);
        free(labels);
        free(centroids);
        return 5;
    }

    /* Запись результатов в CSV */
    rc = csv_write_predictions(out_path, &ds, labels);
    if (rc != 0) { fprintf(stderr, "csv_write_predictions failed: %d\n", rc); return 6; }

    printf("OK\n");
    printf("samples=%d features=%d k=%d inertia=%.10g\n", ds.n_samples, ds.n_features, k, inertia);

    /* Освобождение ресурсов */
    csv_free_dataset(&ds);
    free(labels);
    free(centroids);
    return 0;
}


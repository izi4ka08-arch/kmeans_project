/**
 * @file elbow_demo.c
 * @brief Демонстрационное приложение для метода локтя (elbow method)
 * 
 * Эта программа запускает K-means для диапазона значений k
 * и выводит значения инерции для построения графика "локтя".
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
        "  %s --in data.csv --k-min 2 --k-max 10 [--features 2] [--max-iters 100] [--tol 1e-4] [--seed 42] [--init random|kpp]\n",
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
    int k_min = 2;
    int k_max = 10;
    int n_features = 2;
    int max_iters = 100;
    double tol = 1e-4;
    unsigned int seed = 42;
    kmeans_init_method_t init_method = KMEANS_INIT_KMEANS_PLUS_PLUS;

    /* Парсинг аргументов командной строки */
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (str_eq(a, "--in") && i + 1 < argc) in_path = argv[++i];
        else if (str_eq(a, "--k-min") && i + 1 < argc) k_min = atoi(argv[++i]);
        else if (str_eq(a, "--k-max") && i + 1 < argc) k_max = atoi(argv[++i]);
        else if (str_eq(a, "--max-k") && i + 1 < argc) k_max = atoi(argv[++i]); /* алиас для --k-max */
        else if (str_eq(a, "--features") && i + 1 < argc) n_features = atoi(argv[++i]);
        else if (str_eq(a, "--max-iters") && i + 1 < argc) max_iters = atoi(argv[++i]);
        else if (str_eq(a, "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (str_eq(a, "--seed") && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (str_eq(a, "--init") && i + 1 < argc) {
            const char* v = argv[++i];
            if (str_eq(v, "random")) init_method = KMEANS_INIT_RANDOM;
            else if (str_eq(v, "kpp")) init_method = KMEANS_INIT_KMEANS_PLUS_PLUS;
            else { usage(argv[0]); return 2; }
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    /* Проверка обязательных параметров */
    if (!in_path || k_min <= 0 || k_max < k_min) { usage(argv[0]); return 2; }

    /* Чтение датасета (без меток, так как метод локтя работает без учителя) */
    csv_dataset_t ds;
    int rc = csv_read_dataset(in_path, n_features, -1, &ds);
    if (rc != 0) { fprintf(stderr, "csv_read_dataset failed: %d\n", rc); return 3; }

    /* Выделение памяти для результатов */
    int n_k = k_max - k_min + 1;
    kmeans_elbow_result_t* results = (kmeans_elbow_result_t*)malloc((size_t)n_k * sizeof(kmeans_elbow_result_t));
    if (!results) {
        fprintf(stderr, "out of memory\n");
        csv_free_dataset(&ds);
        return 4;
    }

    /* Запуск метода локтя */
    int n_results = 0;
    rc = kmeans_elbow(
        ds.X, ds.n_samples, ds.n_features,
        k_min, k_max,
        max_iters, tol, seed, init_method,
        results, &n_results
    );
    if (rc != 0) {
        fprintf(stderr, "kmeans_elbow failed: %d\n", rc);
        csv_free_dataset(&ds);
        free(results);
        return 5;
    }

    /* Вывод результатов в формате табуляции */
    printf("# Elbow Method Results\n");
    printf("# k\tinertia\n");
    for (int i = 0; i < n_results; i++) {
        printf("%d\t%.10g\n", results[i].k, results[i].inertia);
    }

    /* Освобождение ресурсов */
    csv_free_dataset(&ds);
    free(results);
    return 0;
}

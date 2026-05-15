/**
 * @file kmeans.c
 * @brief Реализация алгоритма K-means кластеризации
 * 
 * Этот файл содержит реализацию алгоритма K-means с поддержкой:
 * - Случайной инициализации (random)
 * - Умной инициализации k-means++
 * - Метода локтя для подбора оптимального числа кластеров
 */

#include "kmeans.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief XORShift32 генератор псевдослучайных чисел
 * 
 * Быстрый и простой ГСЧ для внутреннего использования.
 * @param state Указатель на состояние генератора (изменяется)
 * @return Случайное 32-битное число
 */
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * @brief Генерирует случайное число в диапазоне [0, 1]
 * 
 * @param state Указатель на состояние ГСЧ
 * @return Двойное число от 0.0 до 1.0
 */
static double rand01(uint32_t* state) {
    return (xorshift32(state) / (double)UINT32_MAX);
}

/**
 * @brief Вычисляет квадрат евклидова расстояния между двумя точками
 * 
 * @param a Первая точка (массив признаков)
 * @param b Вторая точка (массив признаков)
 * @param n_features Количество признаков
 * @return Квадрат евклидова расстояния
 */
static double dist2(const double* a, const double* b, int n_features) {
    double s = 0.0;
    for (int j = 0; j < n_features; j++) {
        double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

/**
 * @brief Копирует координаты точки из одного массива в другой
 * 
 * @param dst Буфер назначения
 * @param src Буфер источника
 * @param n_features Количество признаков для копирования
 */
static void copy_point(double* dst, const double* src, int n_features) {
    memcpy(dst, src, (size_t)n_features * sizeof(double));
}

/**
 * @brief Инициализирует центроиды случайным выбором из данных
 * 
 * Выбирает k случайных точек из набора данных в качестве начальных центроидов.
 * 
 * @param X Входные данные [n_samples × n_features]
 * @param n_samples Количество образцов
 * @param n_features Количество признаков
 * @param k Количество центроидов для инициализации
 * @param rng Указатель на состояние ГСЧ
 * @param centroids_out Выходной массив центроидов [k × n_features]
 * @return 0 при успехе, код ошибки иначе
 */
static int init_random(
    const double* X,
    int n_samples,
    int n_features,
    int k,
    uint32_t* rng,
    double* centroids_out
) {
    if (k > n_samples) return 1;

    int* chosen = (int*)calloc((size_t)n_samples, sizeof(int));
    if (!chosen) return 2;

    for (int c = 0; c < k; c++) {
        int idx = -1;
        /* Пытаемся найти непосещённый индекс случайно */
        for (int tries = 0; tries < 10000; tries++) {
            int r = (int)(rand01(rng) * n_samples);
            if (r < 0) r = 0;
            if (r >= n_samples) r = n_samples - 1;
            if (!chosen[r]) {
                idx = r;
                chosen[r] = 1;
                break;
            }
        }
        /* Если не удалось случайно, берём первый непосещённый */
        if (idx < 0) {
            for (int i = 0; i < n_samples; i++) {
                if (!chosen[i]) {
                    idx = i;
                    chosen[i] = 1;
                    break;
                }
            }
        }
        if (idx < 0) {
            free(chosen);
            return 3;
        }
        copy_point(&centroids_out[c * n_features], &X[idx * n_features], n_features);
    }

    free(chosen);
    return 0;
}

/**
 * @brief Инициализирует центроиды по методу k-means++
 * 
 * K-means++ выбирает начальные центроиды так, чтобы они были
 * далеко друг от друга, что улучшает сходимость алгоритма.
 * 
 * @param X Входные данные [n_samples × n_features]
 * @param n_samples Количество образцов
 * @param n_features Количество признаков
 * @param k Количество центроидов для инициализации
 * @param rng Указатель на состояние ГСЧ
 * @param centroids_out Выходной массив центроидов [k × n_features]
 * @return 0 при успехе, код ошибки иначе
 */
static int init_kmeans_pp(
    const double* X,
    int n_samples,
    int n_features,
    int k,
    uint32_t* rng,
    double* centroids_out
) {
    if (k > n_samples) return 1;

    /* Выбираем первый центроид случайно */
    int first = (int)(rand01(rng) * n_samples);
    if (first < 0) first = 0;
    if (first >= n_samples) first = n_samples - 1;
    copy_point(&centroids_out[0], &X[first * n_features], n_features);

    /* Выделяем память для минимальных квадратов расстояний до ближайшего центроида */
    double* min_d2 = (double*)malloc((size_t)n_samples * sizeof(double));
    if (!min_d2) return 2;

    /* Инициализируем расстояния до первого центроида */
    for (int i = 0; i < n_samples; i++) {
        min_d2[i] = dist2(&X[i * n_features], &centroids_out[0], n_features);
    }

    /* Выбираем остальные k-1 центроидов */
    for (int c = 1; c < k; c++) {
        /* Вычисляем сумму всех расстояний для вероятностного выбора */
        double sum = 0.0;
        for (int i = 0; i < n_samples; i++) sum += min_d2[i];

        int picked = -1;
        /* Обрабатываем вырожденный случай (все расстояния нулевые) */
        if (sum <= 0.0 || !isfinite(sum)) {
            picked = (int)(rand01(rng) * n_samples);
            if (picked < 0) picked = 0;
            if (picked >= n_samples) picked = n_samples - 1;
        } else {
            /* Выбираем точку с вероятностью, пропорциональной d^2 */
            double r = rand01(rng) * sum;
            double acc = 0.0;
            for (int i = 0; i < n_samples; i++) {
                acc += min_d2[i];
                if (acc >= r) {
                    picked = i;
                    break;
                }
            }
            if (picked < 0) picked = n_samples - 1;
        }

        /* Копируем выбранную точку как новый центроид */
        copy_point(&centroids_out[c * n_features], &X[picked * n_features], n_features);

        /* Обновляем минимальные расстояния с учётом нового центроида */
        for (int i = 0; i < n_samples; i++) {
            double d2 = dist2(&X[i * n_features], &centroids_out[c * n_features], n_features);
            if (d2 < min_d2[i]) min_d2[i] = d2;
        }
    }

    free(min_d2);
    return 0;
}

/**
 * @brief Основная функция кластеризации K-means
 * 
 * Выполняет итеративный алгоритм K-means:
 * 1. Инициализация центроидов (random или k-means++)
 * 2. Назначение каждой точки ближайшему центроиду
 * 3. Пересчёт центроидов как среднего точек кластера
 * 4. Повторение шагов 2-3 до сходимости или max_iters
 * 
 * @param X Входные данные [n_samples × n_features], row-major order
 * @param n_samples Количество образцов
 * @param n_features Количество признаков на образец
 * @param params Параметры алгоритма
 * @param labels_out Выходные метки кластеров [n_samples]
 * @param centroids_out Выходные центроиды [k × n_features]
 * @param inertia_out Выходная инерция (сумма квадратов расстояний)
 * @return 0 при успехе, код ошибки иначе
 */
int kmeans_fit(
    const double* X,
    int n_samples,
    int n_features,
    const kmeans_params_t* params,
    int* labels_out,
    double* centroids_out,
    double* inertia_out
) {
    /* Проверка входных параметров */
    if (!X || !params || !labels_out || !centroids_out) return 1;
    if (n_samples <= 0 || n_features <= 0) return 2;
    if (params->k <= 0 || params->k > n_samples) return 3;
    if (params->max_iters <= 0) return 4;
    if (!(params->tol >= 0.0)) return 5;

    const int k = params->k;
    uint32_t rng = (uint32_t)params->seed;
    if (rng == 0u) rng = 0xA341316Cu;  /* Seed по умолчанию */

    /* Инициализация центроидов выбранным методом */
    int init_rc = 0;
    if (params->init_method == KMEANS_INIT_KMEANS_PLUS_PLUS) {
        init_rc = init_kmeans_pp(X, n_samples, n_features, k, &rng, centroids_out);
    } else {
        init_rc = init_random(X, n_samples, n_features, k, &rng, centroids_out);
    }
    if (init_rc != 0) return 10 + init_rc;

    /* Выделение вспомогательной памяти для итераций */
    int* counts = (int*)malloc((size_t)k * sizeof(int));
    double* sums = (double*)malloc((size_t)k * (size_t)n_features * sizeof(double));
    double* old_centroids = (double*)malloc((size_t)k * (size_t)n_features * sizeof(double));
    if (!counts || !sums || !old_centroids) {
        free(counts);
        free(sums);
        free(old_centroids);
        return 20;
    }

    /* Инициализация меток */
    for (int i = 0; i < n_samples; i++) labels_out[i] = -1;

    double inertia = 0.0;
    /* Главный цикл итераций */
    for (int iter = 0; iter < params->max_iters; iter++) {
        int changed = 0;  /* Счётчик изменивших метку точек */
        inertia = 0.0;

        /* Обнуление счётчиков и сумм для каждого кластера */
        for (int c = 0; c < k; c++) {
            counts[c] = 0;
            for (int j = 0; j < n_features; j++) sums[c * n_features + j] = 0.0;
        }

        /* Шаг назначения: находим ближайший центроид для каждой точки */
        for (int i = 0; i < n_samples; i++) {
            const double* xi = &X[i * n_features];
            int best_c = 0;
            double best_d2 = DBL_MAX;
            for (int c = 0; c < k; c++) {
                const double* cc = &centroids_out[c * n_features];
                double d2 = dist2(xi, cc, n_features);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_c = c;
                }
            }

            if (labels_out[i] != best_c) {
                labels_out[i] = best_c;
                changed++;
            }

            inertia += best_d2;
            counts[best_c]++;
            for (int j = 0; j < n_features; j++) sums[best_c * n_features + j] += xi[j];
        }

        /* Сохраняем старые центроиды для проверки сходимости */
        memcpy(old_centroids, centroids_out, (size_t)k * (size_t)n_features * sizeof(double));

        /* Шаг обновления: пересчитываем центроиды */
        for (int c = 0; c < k; c++) {
            if (counts[c] <= 0) {
                /* Пустой кластер: переинициализируем случайной точкой */
                int r = (int)(rand01(&rng) * n_samples);
                if (r < 0) r = 0;
                if (r >= n_samples) r = n_samples - 1;
                copy_point(&centroids_out[c * n_features], &X[r * n_features], n_features);
                continue;
            }
            double inv = 1.0 / (double)counts[c];
            for (int j = 0; j < n_features; j++) {
                centroids_out[c * n_features + j] = sums[c * n_features + j] * inv;
            }
        }

        /* Проверка сходимости: если ничего не изменилось, выходим */
        if (changed == 0) break;

        /* Проверка по порогу смещения центроидов */
        if (params->tol > 0.0) {
            double max_shift = 0.0;
            for (int c = 0; c < k; c++) {
                double shift = sqrt(dist2(&old_centroids[c * n_features], &centroids_out[c * n_features], n_features));
                if (shift > max_shift) max_shift = shift;
            }
            if (max_shift <= params->tol) break;
        }
    }

    /* Освобождение вспомогательной памяти */
    free(counts);
    free(sums);
    free(old_centroids);

    if (inertia_out) *inertia_out = inertia;
    return 0;
}

/**
 * @brief Метод локтя для подбора оптимального числа кластеров
 * 
 * Запускает K-means для каждого значения k от k_min до k_max
 * и возвращает значения инерции. График зависимости инерции от k
 * помогает выбрать оптимальное число кластеров (точка "излома" - "локоть").
 * 
 * @param X Входные данные [n_samples × n_features]
 * @param n_samples Количество образцов
 * @param n_features Количество признаков
 * @param k_min Минимальное значение k для проверки
 * @param k_max Максимальное значение k для проверки
 * @param max_iters Максимум итераций для каждого запуска K-means
 * @param tol Порог сходимости
 * @param seed Seed для генератора случайных чисел
 * @param init_method Метод инициализации центроидов
 * @param results_out Массив результатов [k_max - k_min + 1]
 * @param n_results_out Выходное количество заполненных результатов
 * @return 0 при успехе, код ошибки иначе
 */
int kmeans_elbow(
    const double* X,
    int n_samples,
    int n_features,
    int k_min,
    int k_max,
    int max_iters,
    double tol,
    unsigned int seed,
    kmeans_init_method_t init_method,
    kmeans_elbow_result_t* results_out,
    int* n_results_out
) {
    /* Проверка входных параметров */
    if (!X || !results_out || !n_results_out) return 1;
    if (n_samples <= 0 || n_features <= 0) return 2;
    if (k_min <= 0 || k_max < k_min) return 3;
    if (max_iters <= 0) return 4;
    if (!(tol >= 0.0)) return 5;

    int n_k = k_max - k_min + 1;
    *n_results_out = n_k;

    /* Выделение памяти для промежуточных результатов */
    double* centroids = (double*)malloc((size_t)k_max * (size_t)n_features * sizeof(double));
    int* labels = (int*)malloc((size_t)n_samples * sizeof(int));
    if (!centroids || !labels) {
        free(centroids);
        free(labels);
        return 20;
    }

    /* Запуск K-means для каждого k */
    for (int ki = 0; ki < n_k; ki++) {
        int k = k_min + ki;

        kmeans_params_t params;
        params.k = k;
        params.max_iters = max_iters;
        params.tol = tol;
        params.seed = seed + (unsigned int)ki;  /* Разный seed для каждого k */
        params.init_method = init_method;

        double inertia = 0.0;
        int rc = kmeans_fit(X, n_samples, n_features, &params, labels, centroids, &inertia);
        if (rc != 0) {
            free(centroids);
            free(labels);
            return 100 + rc;
        }

        results_out[ki].k = k;
        results_out[ki].inertia = inertia;
    }

    /* Освобождение памяти */
    free(centroids);
    free(labels);
    return 0;
}


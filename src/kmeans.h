#pragma once

/**
 * @file kmeans.h
 * @brief Заголовочный файл библиотеки K-means кластеризации
 * 
 * Эта библиотека предоставляет функции для выполнения алгоритма K-means
 * и метода локтя (elbow method) для подбора оптимального числа кластеров.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Перечисление методов инициализации центроидов
 * 
 * KMEANS_INIT_RANDOM - случайная инициализация из набора данных
 * KMEANS_INIT_KMEANS_PLUS_PLUS - умная инициализация k-means++ для лучшей сходимости
 */
typedef enum kmeans_init_method {
    KMEANS_INIT_RANDOM = 0,           /**< Случайная инициализация */
    KMEANS_INIT_KMEANS_PLUS_PLUS = 1  /**< Инициализация k-means++ */
} kmeans_init_method_t;

/**
 * @brief Структура параметров для алгоритма K-means
 */
typedef struct kmeans_params {
    int k;                              /**< Количество кластеров */
    int max_iters;                      /**< Максимальное количество итераций */
    double tol;                         /**< Порог сходимости (максимальное смещение центроидов) */
    unsigned int seed;                  /**< Seed для генератора случайных чисел */
    kmeans_init_method_t init_method;   /**< Метод инициализации центроидов */
} kmeans_params_t;

/**
 * @brief Выполняет кластеризацию K-means на входных данных
 * 
 * @param X Входные данные размером [n_samples × n_features], row-major order
 * @param n_samples Количество образцов в наборе данных
 * @param n_features Количество признаков на образец
 * @param params Параметры алгоритма (k, max_iters, tol, seed, init_method)
 * @param labels_out Выходной массив меток кластеров размером [n_samples]
 * @param centroids_out Выходной массив центроидов размером [k × n_features]
 * @param inertia_out Выходное значение инерции (сумма квадратов расстояний до центроидов)
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
);

/**
 * @brief Структура результата для метода локтя
 */
typedef struct kmeans_elbow_result {
    int k;          /**< Количество кластеров */
    double inertia; /**< Значение инерции для данного k */
} kmeans_elbow_result_t;

/**
 * @brief Выполняет метод локтя для подбора оптимального числа кластеров
 * 
 * Запускает K-means для каждого k от k_min до k_max и возвращает
 * значения инерции для построения графика "локтя".
 * 
 * @param X Входные данные размером [n_samples × n_features]
 * @param n_samples Количество образцов
 * @param n_features Количество признаков
 * @param k_min Минимальное значение k для проверки
 * @param k_max Максимальное значение k для проверки
 * @param max_iters Максимум итераций для каждого запуска K-means
 * @param tol Порог сходимости
 * @param seed Seed для генератора случайных чисел
 * @param init_method Метод инициализации
 * @param results_out Массив результатов размером [k_max - k_min + 1]
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
);

#ifdef __cplusplus
}
#endif


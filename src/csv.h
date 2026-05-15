/**
 * @file csv.h
 * @brief Заголовочный файл для работы с CSV-файлами
 * 
 * Предоставляет функции для чтения датасетов из CSV и записи результатов.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Структура, представляющая загруженный датасет
 */
typedef struct csv_dataset {
    int n_samples;      /**< Количество образцов (строк) в датасете */
    int n_features;     /**< Количество признаков (столбцов) */
    double* X;          /**< Массив данных размером [n_samples × n_features] */
    int* y_true;        /**< Массив истинных меток (может быть NULL) */
} csv_dataset_t;

/**
 * @brief Читает датасет из CSV-файла
 * 
 * @param path Путь к CSV-файлу
 * @param n_features Ожидаемое количество признаков (столбцов с данными)
 * @param label_col Индекс столбца с метками (-1 если нет меток)
 * @param out Выходная структура датасета
 * @return 0 при успехе, код ошибки иначе
 */
int csv_read_dataset(
    const char* path,
    int n_features,
    int label_col,
    csv_dataset_t* out
);

/**
 * @brief Освобождает память, занятую структурой датасета
 * 
 * @param ds Указатель на структуру датасета
 */
void csv_free_dataset(csv_dataset_t* ds);

/**
 * @brief Записывает предсказанные метки в CSV-файл
 * 
 * @param path Путь к выходному файлу
 * @param ds Исходный датасет (данные и опционально истинные метки)
 * @param pred_labels Массив предсказанных меток размером [n_samples]
 * @return 0 при успехе, код ошибки иначе
 */
int csv_write_predictions(
    const char* path,
    const csv_dataset_t* ds,
    const int* pred_labels
);

#ifdef __cplusplus
}
#endif


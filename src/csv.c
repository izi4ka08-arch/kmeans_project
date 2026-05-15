/**
 * @file csv.c
 * @brief Реализация функций для чтения/записи CSV-файлов
 * 
 * Этот файл содержит функции для:
 * - Чтения датасетов из CSV-файлов
 * - Освобождения памяти датасета
 * - Записи предсказанных меток в CSV
 */

#include "csv.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Инициализирует структуру датасета нулевыми значениями
 * 
 * @param ds Указатель на структуру для инициализации
 */
static void ds_init(csv_dataset_t* ds) {
    ds->n_samples = 0;
    ds->n_features = 0;
    ds->X = NULL;
    ds->y_true = NULL;
}

/**
 * @brief Освобождает всю память, занятую структурой датасета
 * 
 * @param ds Указатель на структуру датасета
 */
void csv_free_dataset(csv_dataset_t* ds) {
    if (!ds) return;
    free(ds->X);
    free(ds->y_true);
    ds_init(ds);
}

/**
 * @brief Удаляет ведущие пробельные символы из строки
 * 
 * @param s Входная строка
 * @return Указатель на первый непробельный символ
 */
static char* trim_left(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

/**
 * @brief Удаляет trailing пробельные символы из строки (на месте)
 * 
 * @param s Строка для обработки (изменяется)
 */
static void trim_right_inplace(char* s) {
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) {
        s[n - 1] = '\0';
        n--;
    }
}

/**
 * @brief Разбивает CSV-строку на поля по разделителю ','
 * 
 * @param line Строка для разбора (изменяется, поля заменяются на \0)
 * @param fields Массив указателей на поля (заполняется функцией)
 * @param max_fields Максимальное количество полей для извлечения
 * @return Количество найденных полей
 */
static int split_csv_inplace(char* line, char** fields, int max_fields) {
    int n = 0;
    char* p = line;
    while (*p && n < max_fields) {
        fields[n++] = p;
        while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
        if (*p == ',') {
            *p = '\0';
            p++;
        } else {
            if (*p) *p = '\0';
            break;
        }
    }
    return n;
}

/**
 * @brief Читает датасет из CSV-файла
 * 
 * Функция читает CSV-файл, пропускает заголовок и загружает числовые данные.
 * Поддерживает опциональное чтение меток из указанного столбца.
 * 
 * @param path Путь к CSV-файлу
 * @param n_features Количество признаков (столбцов с данными)
 * @param label_col Индекс столбца с метками (-1 если меток нет)
 * @param out Выходная структура датасета
 * @return 0 при успехе, код ошибки иначе (1-12)
 */
int csv_read_dataset(
    const char* path,
    int n_features,
    int label_col,
    csv_dataset_t* out
) {
    /* Проверка входных параметров */
    if (!path || !out) return 1;
    if (n_features <= 0) return 2;
    if (label_col >= 0 && label_col < n_features) return 3;

    FILE* f = fopen(path, "rb");
    if (!f) return 4;

    ds_init(out);
    out->n_features = n_features;

    /* Чтение и пропуск строки заголовка */
    char buf[8192];
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        return 5;
    }

    /* Выделение начального буфера для данных */
    int cap = 256;
    double* X = (double*)malloc((size_t)cap * (size_t)n_features * sizeof(double));
    int* y = NULL;
    if (label_col >= 0) y = (int*)malloc((size_t)cap * sizeof(int));
    if (!X || (label_col >= 0 && !y)) {
        fclose(f);
        free(X);
        free(y);
        return 6;
    }

    int n = 0;
    /* Чтение строк данных */
    while (fgets(buf, sizeof(buf), f)) {
        char* line = trim_left(buf);
        trim_right_inplace(line);
        if (*line == '\0') continue;  /* Пропуск пустых строк */

        char* fields[256];
        int nf = split_csv_inplace(line, fields, 256);
        if (nf < n_features) {
            fclose(f);
            free(X);
            free(y);
            return 7;  /* Недостаточно столбцов */
        }
        if (label_col >= 0 && nf <= label_col) {
            fclose(f);
            free(X);
            free(y);
            return 8;  /* Столбец меток выходит за границы */
        }

        /* Увеличение ёмкости при необходимости */
        if (n >= cap) {
            int new_cap = cap * 2;
            double* X2 = (double*)realloc(X, (size_t)new_cap * (size_t)n_features * sizeof(double));
            int* y2 = y ? (int*)realloc(y, (size_t)new_cap * sizeof(int)) : NULL;
            if (!X2 || (y && !y2)) {
                fclose(f);
                free(X2 ? X2 : X);
                free(y2 ? y2 : y);
                return 9;  /* Ошибка realloc */
            }
            X = X2;
            y = y2;
            cap = new_cap;
        }

        /* Парсинг признаков */
        for (int j = 0; j < n_features; j++) {
            char* endp = NULL;
            errno = 0;
            double v = strtod(fields[j], &endp);
            if (errno != 0 || endp == fields[j]) {
                fclose(f);
                free(X);
                free(y);
                return 10;  /* Ошибка парсинга числа */
            }
            X[(size_t)n * (size_t)n_features + (size_t)j] = v;
        }

        /* Парсинг метки если есть */
        if (label_col >= 0) {
            char* endp = NULL;
            long lv = strtol(fields[label_col], &endp, 10);
            if (endp == fields[label_col]) {
                fclose(f);
                free(X);
                free(y);
                return 11;  /* Ошибка парсинга метки */
            }
            y[n] = (int)lv;
        }

        n++;
    }

    fclose(f);

    /* Финализация структуры */
    out->n_samples = n;
    out->X = (double*)realloc(X, (size_t)n * (size_t)n_features * sizeof(double));
    out->y_true = y ? (int*)realloc(y, (size_t)n * sizeof(int)) : NULL;
    if (!out->X) {
        csv_free_dataset(out);
        return 12;
    }
    return 0;
}

/**
 * @brief Записывает предсказанные метки в CSV-файл
 * 
 * Создаёт CSV-файл с исходными данными, истинными метками (если есть)
 * и предсказанными метками.
 * 
 * @param path Путь к выходному файлу
 * @param ds Исходный датасет
 * @param pred_labels Массив предсказанных меток [n_samples]
 * @return 0 при успехе, код ошибки иначе
 */
int csv_write_predictions(
    const char* path,
    const csv_dataset_t* ds,
    const int* pred_labels
) {
    if (!path || !ds || !pred_labels) return 1;
    if (ds->n_samples <= 0 || ds->n_features <= 0 || !ds->X) return 2;

    FILE* f = fopen(path, "wb");
    if (!f) return 3;

    /* Запись заголовка */
    for (int j = 0; j < ds->n_features; j++) fprintf(f, "x%d,", j + 1);
    if (ds->y_true) fprintf(f, "true_label,");
    fprintf(f, "pred_label\n");

    /* Запись данных */
    for (int i = 0; i < ds->n_samples; i++) {
        for (int j = 0; j < ds->n_features; j++) {
            double v = ds->X[(size_t)i * (size_t)ds->n_features + (size_t)j];
            fprintf(f, "%.17g,", v);
        }
        if (ds->y_true) fprintf(f, "%d,", ds->y_true[i]);
        fprintf(f, "%d\n", pred_labels[i]);
    }

    fclose(f);
    return 0;
}


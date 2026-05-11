#include "csv.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ds_init(csv_dataset_t* ds) {
    ds->n_samples = 0;
    ds->n_features = 0;
    ds->X = NULL;
    ds->y_true = NULL;
}

void csv_free_dataset(csv_dataset_t* ds) {
    if (!ds) return;
    free(ds->X);
    free(ds->y_true);
    ds_init(ds);
}

static char* trim_left(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

static void trim_right_inplace(char* s) {
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) {
        s[n - 1] = '\0';
        n--;
    }
}

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

int csv_read_dataset(
    const char* path,
    int n_features,
    int label_col,
    csv_dataset_t* out
) {
    if (!path || !out) return 1;
    if (n_features <= 0) return 2;
    if (label_col >= 0 && label_col < n_features) return 3;

    FILE* f = fopen(path, "rb");
    if (!f) return 4;

    ds_init(out);
    out->n_features = n_features;

    char buf[8192];
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        return 5;
    }

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
    while (fgets(buf, sizeof(buf), f)) {
        char* line = trim_left(buf);
        trim_right_inplace(line);
        if (*line == '\0') continue;

        char* fields[256];
        int nf = split_csv_inplace(line, fields, 256);
        if (nf < n_features) {
            fclose(f);
            free(X);
            free(y);
            return 7;
        }
        if (label_col >= 0 && nf <= label_col) {
            fclose(f);
            free(X);
            free(y);
            return 8;
        }

        if (n >= cap) {
            int new_cap = cap * 2;
            double* X2 = (double*)realloc(X, (size_t)new_cap * (size_t)n_features * sizeof(double));
            int* y2 = y ? (int*)realloc(y, (size_t)new_cap * sizeof(int)) : NULL;
            if (!X2 || (y && !y2)) {
                fclose(f);
                free(X2 ? X2 : X);
                free(y2 ? y2 : y);
                return 9;
            }
            X = X2;
            y = y2;
            cap = new_cap;
        }

        for (int j = 0; j < n_features; j++) {
            char* endp = NULL;
            errno = 0;
            double v = strtod(fields[j], &endp);
            if (errno != 0 || endp == fields[j]) {
                fclose(f);
                free(X);
                free(y);
                return 10;
            }
            X[(size_t)n * (size_t)n_features + (size_t)j] = v;
        }

        if (label_col >= 0) {
            char* endp = NULL;
            long lv = strtol(fields[label_col], &endp, 10);
            if (endp == fields[label_col]) {
                fclose(f);
                free(X);
                free(y);
                return 11;
            }
            y[n] = (int)lv;
        }

        n++;
    }

    fclose(f);

    out->n_samples = n;
    out->X = (double*)realloc(X, (size_t)n * (size_t)n_features * sizeof(double));
    out->y_true = y ? (int*)realloc(y, (size_t)n * sizeof(int)) : NULL;
    if (!out->X) {
        csv_free_dataset(out);
        return 12;
    }
    return 0;
}

int csv_write_predictions(
    const char* path,
    const csv_dataset_t* ds,
    const int* pred_labels
) {
    if (!path || !ds || !pred_labels) return 1;
    if (ds->n_samples <= 0 || ds->n_features <= 0 || !ds->X) return 2;

    FILE* f = fopen(path, "wb");
    if (!f) return 3;

    for (int j = 0; j < ds->n_features; j++) fprintf(f, "x%d,", j + 1);
    if (ds->y_true) fprintf(f, "true_label,");
    fprintf(f, "pred_label\n");

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


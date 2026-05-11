#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct csv_dataset {
    int n_samples;
    int n_features;
    double* X;
    int* y_true; /* необязательно, может быть NULL */
} csv_dataset_t;

int csv_read_dataset(
    const char* path,
    int n_features,
    int label_col,
    csv_dataset_t* out
);

void csv_free_dataset(csv_dataset_t* ds);

int csv_write_predictions(
    const char* path,
    const csv_dataset_t* ds,
    const int* pred_labels
);

#ifdef __cplusplus
}
#endif


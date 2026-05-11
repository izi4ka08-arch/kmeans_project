#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kmeans_init_method {
    KMEANS_INIT_RANDOM = 0,
    KMEANS_INIT_KMEANS_PLUS_PLUS = 1
} kmeans_init_method_t;

typedef struct kmeans_params {
    int k;
    int max_iters;
    double tol;
    unsigned int seed;
    kmeans_init_method_t init_method;
} kmeans_params_t;

int kmeans_fit(
    const double* X,
    int n_samples,
    int n_features,
    const kmeans_params_t* params,
    int* labels_out,
    double* centroids_out,
    double* inertia_out
);

#ifdef __cplusplus
}
#endif


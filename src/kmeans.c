#include "kmeans.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double rand01(uint32_t* state) {
    return (xorshift32(state) / (double)UINT32_MAX);
}

static double dist2(const double* a, const double* b, int n_features) {
    double s = 0.0;
    for (int j = 0; j < n_features; j++) {
        double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

static void copy_point(double* dst, const double* src, int n_features) {
    memcpy(dst, src, (size_t)n_features * sizeof(double));
}

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

static int init_kmeans_pp(
    const double* X,
    int n_samples,
    int n_features,
    int k,
    uint32_t* rng,
    double* centroids_out
) {
    if (k > n_samples) return 1;

    int first = (int)(rand01(rng) * n_samples);
    if (first < 0) first = 0;
    if (first >= n_samples) first = n_samples - 1;
    copy_point(&centroids_out[0], &X[first * n_features], n_features);

    double* min_d2 = (double*)malloc((size_t)n_samples * sizeof(double));
    if (!min_d2) return 2;

    for (int i = 0; i < n_samples; i++) {
        min_d2[i] = dist2(&X[i * n_features], &centroids_out[0], n_features);
    }

    for (int c = 1; c < k; c++) {
        double sum = 0.0;
        for (int i = 0; i < n_samples; i++) sum += min_d2[i];

        int picked = -1;
        if (sum <= 0.0 || !isfinite(sum)) {
            picked = (int)(rand01(rng) * n_samples);
            if (picked < 0) picked = 0;
            if (picked >= n_samples) picked = n_samples - 1;
        } else {
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

        copy_point(&centroids_out[c * n_features], &X[picked * n_features], n_features);

        for (int i = 0; i < n_samples; i++) {
            double d2 = dist2(&X[i * n_features], &centroids_out[c * n_features], n_features);
            if (d2 < min_d2[i]) min_d2[i] = d2;
        }
    }

    free(min_d2);
    return 0;
}

int kmeans_fit(
    const double* X,
    int n_samples,
    int n_features,
    const kmeans_params_t* params,
    int* labels_out,
    double* centroids_out,
    double* inertia_out
) {
    if (!X || !params || !labels_out || !centroids_out) return 1;
    if (n_samples <= 0 || n_features <= 0) return 2;
    if (params->k <= 0 || params->k > n_samples) return 3;
    if (params->max_iters <= 0) return 4;
    if (!(params->tol >= 0.0)) return 5;

    const int k = params->k;
    uint32_t rng = (uint32_t)params->seed;
    if (rng == 0u) rng = 0xA341316Cu;

    int init_rc = 0;
    if (params->init_method == KMEANS_INIT_KMEANS_PLUS_PLUS) {
        init_rc = init_kmeans_pp(X, n_samples, n_features, k, &rng, centroids_out);
    } else {
        init_rc = init_random(X, n_samples, n_features, k, &rng, centroids_out);
    }
    if (init_rc != 0) return 10 + init_rc;

    int* counts = (int*)malloc((size_t)k * sizeof(int));
    double* sums = (double*)malloc((size_t)k * (size_t)n_features * sizeof(double));
    double* old_centroids = (double*)malloc((size_t)k * (size_t)n_features * sizeof(double));
    if (!counts || !sums || !old_centroids) {
        free(counts);
        free(sums);
        free(old_centroids);
        return 20;
    }

    for (int i = 0; i < n_samples; i++) labels_out[i] = -1;

    double inertia = 0.0;
    for (int iter = 0; iter < params->max_iters; iter++) {
        int changed = 0;
        inertia = 0.0;

        for (int c = 0; c < k; c++) {
            counts[c] = 0;
            for (int j = 0; j < n_features; j++) sums[c * n_features + j] = 0.0;
        }

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

        memcpy(old_centroids, centroids_out, (size_t)k * (size_t)n_features * sizeof(double));

        for (int c = 0; c < k; c++) {
            if (counts[c] <= 0) {
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

        if (changed == 0) break;

        if (params->tol > 0.0) {
            double max_shift = 0.0;
            for (int c = 0; c < k; c++) {
                double shift = sqrt(dist2(&old_centroids[c * n_features], &centroids_out[c * n_features], n_features));
                if (shift > max_shift) max_shift = shift;
            }
            if (max_shift <= params->tol) break;
        }
    }

    free(counts);
    free(sums);
    free(old_centroids);

    if (inertia_out) *inertia_out = inertia;
    return 0;
}


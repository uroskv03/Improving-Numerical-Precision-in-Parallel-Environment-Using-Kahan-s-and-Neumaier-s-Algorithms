#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <quadmath.h>

typedef struct {
    double sum;
    double corr;
} KahanAcc;

static inline KahanAcc kahan_update(KahanAcc acc, double value) {
    double y = value - acc.corr;
    double t = acc.sum + y;
    acc.corr = (t - acc.sum) - y;
    acc.sum = t;
    return acc;
}

static inline KahanAcc neumaier_update(KahanAcc acc, double value) {
    double t = acc.sum + value;
    if (fabs(acc.sum) >= fabs(value)) {
        acc.corr += (acc.sum - t) + value;
    } else {
        acc.corr += (value - t) + acc.sum;
    }
    acc.sum = t;
    return acc;
}

#pragma omp declare reduction(kahanRed : KahanAcc : \
    omp_out = kahan_update(omp_out, omp_in.sum), \
    omp_out.corr += omp_in.corr) \
    initializer(omp_priv = {0.0, 0.0})

#pragma omp declare reduction(neumaierRed : KahanAcc : \
    omp_out = neumaier_update(omp_out, omp_in.sum), \
    omp_out.corr += omp_in.corr) \
    initializer(omp_priv = {0.0, 0.0})

static void fill_random(double *X, long N, double min, double max) {
    for (long i = 0; i < N; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        X[i] = min + r * (max - min);
    }
}

static void fill_constant_random(double *X, long N) {
    double v = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    for (long i = 0; i < N; i++) {
        X[i] = v;
    }
}

static void fill_constant_value(double *X, long N, double value) {
    for (long i = 0; i < N; i++) {
        X[i] = value;
    }
}

static void fill_matrix_random(double *M, long rows, long cols, double min, double max) {
    for (long i = 0; i < rows * cols; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        M[i] = min + r * (max - min);
    }
}

static void fill_matrix_constant_random(double *M, long rows, long cols) {
    double v = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    for (long i = 0; i < rows * cols; i++) {
        M[i] = v;
    }
}

static void fill_matrix_constant_value(double *M, long rows, long cols, double value) {
    for (long i = 0; i < rows * cols; i++) {
        M[i] = value;
    }
}

static int load_matrix_from_file(const char *path, double **outA, double **outB, long *outN) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    long N = 0;
    if (fscanf(f, "%ld", &N) != 1) {
        fclose(f);
        return 0;
    }

    double *A = (double*)malloc((size_t)N * N * sizeof(double));
    double *B = (double*)malloc((size_t)N * N * sizeof(double));
    if (!A || !B) {
        free(A);
        free(B);
        fclose(f);
        return 0;
    }

    for (long i = 0; i < N * N; i++) {
        if (fscanf(f, "%lf", &A[i]) != 1) {
            free(A);
            free(B);
            fclose(f);
            return 0;
        }
    }

    for (long i = 0; i < N * N; i++) {
        if (fscanf(f, "%lf", &B[i]) != 1) {
            free(A);
            free(B);
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    *outA = A;
    *outB = B;
    *outN = N;
    return 1;
}

static void matmul_native(const double *A, const double *B, double *C, long N) {
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            double sum = 0.0;
            for (long k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void matmul_kahan(const double *A, const double *B, double *C, long N) {
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            KahanAcc acc = {0.0, 0.0};
            for (long k = 0; k < N; k++) {
                acc = kahan_update(acc, A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = acc.sum + acc.corr;
        }
    }
}

static void matmul_neumaier(const double *A, const double *B, double *C, long N) {
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            KahanAcc acc = {0.0, 0.0};
            for (long k = 0; k < N; k++) {
                acc = neumaier_update(acc, A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = acc.sum + acc.corr;
        }
    }
}

static void matmul_parallel_native(const double *A, const double *B, double *C, long N) {
    #pragma omp parallel for collapse(2)
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            double sum = 0.0;
            for (long k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void matmul_parallel_kahan(const double *A, const double *B, double *C, long N) {
    #pragma omp parallel for collapse(2)
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            KahanAcc acc = {0.0, 0.0};
            for (long k = 0; k < N; k++) {
                acc = kahan_update(acc, A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = acc.sum + acc.corr;
        }
    }
}

static void matmul_parallel_neumaier(const double *A, const double *B, double *C, long N) {
    #pragma omp parallel for collapse(2)
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            KahanAcc acc = {0.0, 0.0};
            for (long k = 0; k < N; k++) {
                acc = neumaier_update(acc, A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = acc.sum + acc.corr;
        }
    }
}

static void matmul_reference_quad(const double *A, const double *B, __float128 *Cref, long N) {
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            __float128 sum = 0.0Q;
            for (long k = 0; k < N; k++) {
                sum += (__float128)A[i * N + k] * (__float128)B[k * N + j];
            }
            Cref[i * N + j] = sum;
        }
    }
}

static double matrix_absolute_error(const double *reference, const double *value, long N) {
    double err = 0.0;
    long count = N * N;
    for (long idx = 0; idx < count; idx++) {
        err += fabs(reference[idx] - value[idx]);
    }
    return err;
}

static double matrix_relative_error(const double *reference, const double *value, long N) {
    double num = 0.0;
    double den = 0.0;
    long count = N * N;
    for (long idx = 0; idx < count; idx++) {
        num += fabs(reference[idx] - value[idx]);
        den += fabs(reference[idx]);
    }
    if (den == 0.0) return 0.0;
    return num / den;
}

static long double matrix_quad_absolute_error(const __float128 *reference, const double *value, long N) {
    long double err = 0.0L;
    long count = N * N;
    for (long idx = 0; idx < count; idx++) {
        err += fabsl((long double)reference[idx] - (long double)value[idx]);
    }
    return err;
}

static long double matrix_quad_relative_error(const __float128 *reference, const double *value, long N) {
    long double num = 0.0L;
    long double den = 0.0L;
    long count = N * N;
    for (long idx = 0; idx < count; idx++) {
        num += fabsl((long double)reference[idx] - (long double)value[idx]);
        den += fabsl((long double)reference[idx]);
    }
    if (den == 0.0L) return 0.0L;
    return num / den;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N|file> [random|constant|value]\n", argv[0]);
        return 1;
    }

    long N = 0;
    double *A = NULL;
    double *B = NULL;

    if (argc >= 2 && fopen(argv[1], "r") != NULL) {
        if (!load_matrix_from_file(argv[1], &A, &B, &N)) {
            fprintf(stderr, "Neuspesno citanje iz fajla: %s\n", argv[1]);
            return 1;
        }
    } else {
        N = strtol(argv[1], NULL, 10);
        if (N <= 0) return 1;

        A = (double*)malloc((size_t)N * N * sizeof(double));
        B = (double*)malloc((size_t)N * N * sizeof(double));
        if (!A || !B) {
            free(A);
            free(B);
            return 1;
        }

        if (argc >= 3) {
            if (strcmp(argv[2], "random") == 0) {
                fill_matrix_random(A, N, N, -10.0, 10.0);
                fill_matrix_random(B, N, N, -10.0, 10.0);
            } else if (strcmp(argv[2], "constant") == 0) {
                fill_matrix_constant_random(A, N, N);
                fill_matrix_constant_random(B, N, N);
            } else if (strcmp(argv[2], "value") == 0) {
                fill_matrix_constant_value(A, N, N, 0.1);
                fill_matrix_constant_value(B, N, N, 0.1);
            } else {
                fprintf(stderr, "Nepoznat tip generisanja: %s\n", argv[2]);
                free(A);
                free(B);
                return 1;
            }
        } else {
            fill_matrix_random(A, N, N, -10.0, 10.0);
            fill_matrix_random(B, N, N, -10.0, 10.0);
        }
    }

    long count = N * N;
    double *C_native = (double*)malloc((size_t)count * sizeof(double));
    double *C_kahan = (double*)malloc((size_t)count * sizeof(double));
    double *C_neum = (double*)malloc((size_t)count * sizeof(double));
    double *C_p_native = (double*)malloc((size_t)count * sizeof(double));
    double *C_p_kahan = (double*)malloc((size_t)count * sizeof(double));
    double *C_p_neum = (double*)malloc((size_t)count * sizeof(double));
    __float128 *C_ref_q = (__float128*)malloc((size_t)count * sizeof(__float128));
    double *C_ref_d = (double*)malloc((size_t)count * sizeof(double));

    if (!C_native || !C_kahan || !C_neum || !C_p_native || !C_p_kahan || !C_p_neum || !C_ref_q || !C_ref_d) {
        free(A); free(B);
        free(C_native); free(C_kahan); free(C_neum); free(C_p_native); free(C_p_kahan); free(C_p_neum);
        free(C_ref_q); free(C_ref_d);
        return 1;
    }

    matmul_reference_quad(A, B, C_ref_q, N);
    for (long idx = 0; idx < count; idx++) {
        C_ref_d[idx] = (double)C_ref_q[idx];
    }

    printf("N: %ld\n", N);
    printf("Quad reference matrix computed.\n");
    printf("%-18s | %-18s | %-18s | %-18s | %-18s | %-18s | %-8s\n",
           "Algoritam", "AbsErr(Double)", "RelErr(Double)", "AbsErr(Quad)", "RelErr(Quad)", "MaxDiff", "Vreme");
    printf("------------------+------------------+------------------+------------------+------------------+------------------+----------\n");

    double t0 = 0.0;
    double elapsed = 0.0;

    t0 = omp_get_wtime();
    matmul_native(A, B, C_native, N);
    elapsed = omp_get_wtime() - t0;
    double max_native = 0.0;
    for (long idx = 0; idx < count; idx++) {
        double d = fabs(C_ref_d[idx] - C_native[idx]);
        if (d > max_native) max_native = d;
    }
    printf("%-18s | %-18e | %-18e | %-18Le | %-18Le | %-18e | %-8.4fs\n",
           "Serial Native",
           matrix_absolute_error(C_ref_d, C_native, N),
           matrix_relative_error(C_ref_d, C_native, N),
           matrix_quad_absolute_error(C_ref_q, C_native, N),
           matrix_quad_relative_error(C_ref_q, C_native, N),
           max_native,
           elapsed);

    t0 = omp_get_wtime();
    matmul_kahan(A, B, C_kahan, N);
    elapsed = omp_get_wtime() - t0;
    double max_kahan = 0.0;
    for (long idx = 0; idx < count; idx++) {
        double d = fabs(C_ref_d[idx] - C_kahan[idx]);
        if (d > max_kahan) max_kahan = d;
    }
    printf("%-18s | %-18e | %-18e | %-18Le | %-18Le | %-18e | %-8.4fs\n",
           "Serial Kahan",
           matrix_absolute_error(C_ref_d, C_kahan, N),
           matrix_relative_error(C_ref_d, C_kahan, N),
           matrix_quad_absolute_error(C_ref_q, C_kahan, N),
           matrix_quad_relative_error(C_ref_q, C_kahan, N),
           max_kahan,
           elapsed);

    t0 = omp_get_wtime();
    matmul_neumaier(A, B, C_neum, N);
    elapsed = omp_get_wtime() - t0;
    double max_neum = 0.0;
    for (long idx = 0; idx < count; idx++) {
        double d = fabs(C_ref_d[idx] - C_neum[idx]);
        if (d > max_neum) max_neum = d;
    }
    printf("%-18s | %-18e | %-18e | %-18Le | %-18Le | %-18e | %-8.4fs\n",
           "Serial Neumaier",
           matrix_absolute_error(C_ref_d, C_neum, N),
           matrix_relative_error(C_ref_d, C_neum, N),
           matrix_quad_absolute_error(C_ref_q, C_neum, N),
           matrix_quad_relative_error(C_ref_q, C_neum, N),
           max_neum,
           elapsed);

    t0 = omp_get_wtime();
    matmul_parallel_native(A, B, C_p_native, N);
    elapsed = omp_get_wtime() - t0;
    double max_p_native = 0.0;
    for (long idx = 0; idx < count; idx++) {
        double d = fabs(C_ref_d[idx] - C_p_native[idx]);
        if (d > max_p_native) max_p_native = d;
    }
    printf("%-18s | %-18e | %-18e | %-18Le | %-18Le | %-18e | %-8.4fs\n",
           "Parallel Native",
           matrix_absolute_error(C_ref_d, C_p_native, N),
           matrix_relative_error(C_ref_d, C_p_native, N),
           matrix_quad_absolute_error(C_ref_q, C_p_native, N),
           matrix_quad_relative_error(C_ref_q, C_p_native, N),
           max_p_native,
           elapsed);

    t0 = omp_get_wtime();
    matmul_parallel_kahan(A, B, C_p_kahan, N);
    elapsed = omp_get_wtime() - t0;
    double max_p_kahan = 0.0;
    for (long idx = 0; idx < count; idx++) {
        double d = fabs(C_ref_d[idx] - C_p_kahan[idx]);
        if (d > max_p_kahan) max_p_kahan = d;
    }
    printf("%-18s | %-18e | %-18e | %-18Le | %-18Le | %-18e | %-8.4fs\n",
           "Parallel Kahan",
           matrix_absolute_error(C_ref_d, C_p_kahan, N),
           matrix_relative_error(C_ref_d, C_p_kahan, N),
           matrix_quad_absolute_error(C_ref_q, C_p_kahan, N),
           matrix_quad_relative_error(C_ref_q, C_p_kahan, N),
           max_p_kahan,
           elapsed);

    t0 = omp_get_wtime();
    matmul_parallel_neumaier(A, B, C_p_neum, N);
    elapsed = omp_get_wtime() - t0;
    double max_p_neum = 0.0;
    for (long idx = 0; idx < count; idx++) {
        double d = fabs(C_ref_d[idx] - C_p_neum[idx]);
        if (d > max_p_neum) max_p_neum = d;
    }
    printf("%-18s | %-18e | %-18e | %-18Le | %-18Le | %-18e | %-8.4fs\n",
           "Parallel Neumaier",
           matrix_absolute_error(C_ref_d, C_p_neum, N),
           matrix_relative_error(C_ref_d, C_p_neum, N),
           matrix_quad_absolute_error(C_ref_q, C_p_neum, N),
           matrix_quad_relative_error(C_ref_q, C_p_neum, N),
           max_p_neum,
           elapsed);

    free(A);
    free(B);
    free(C_native); free(C_kahan); free(C_neum); free(C_p_native); free(C_p_kahan); free(C_p_neum);
    free(C_ref_q); free(C_ref_d);
    return 0;
}
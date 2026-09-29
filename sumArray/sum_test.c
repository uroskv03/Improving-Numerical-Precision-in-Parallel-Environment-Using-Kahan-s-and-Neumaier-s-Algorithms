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
    double v = (double)rand() / (double)RAND_MAX * 2.0 - 1.0; // [-1,1]
    for (long i = 0; i < N; i++) {
        X[i] = v;
    }
}

static void fill_constant_value(double *X, long N, double value) {
    for (long i = 0; i < N; i++) {
        X[i] = value;
    }
}

static int load_from_file(const char *path, double **outX, long *outN) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    long N = 0;
    if (fscanf(f, "%ld", &N) != 1) {
        fclose(f);
        return 0;
    }

    double *X = (double*)malloc((size_t)N * sizeof(double));
    if (!X) {
        fclose(f);
        return 0;
    }

    for (long i = 0; i < N; i++) {
        if (fscanf(f, "%lf", &X[i]) != 1) {
            free(X);
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    *outX = X;
    *outN = N;
    return 1;
}

static double seq_native(double *X, long N) {
    double s = 0.0;
    for (long i = 0; i < N; i++) s += X[i];
    return s;
}

static double seq_kahan(double *X, long N) {
    double s = X[0], c = 0.0;
    for (long i = 1; i < N; i++) {
        double y = X[i] - c;
        double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

static double seq_neumaier(double *X, long N) {
    double s = X[0];
    double c = 0.0;
    for (long i = 1; i < N; i++) {
        double t = s + X[i];
        if (fabs(s) >= fabs(X[i])) {
            c += (s - t) + X[i];
        } else {
            c += (X[i] - t) + s;
        }
        s = t;
    }
    return s + c;
}

static double absolute_error(double reference, double value) {
    return fabs(reference - value);
}

static double relative_error(double reference, double value) {
    double denom = fmax(fabs(reference), fabs(value));
    if (denom == 0.0) return 0.0;
    return fabs(reference - value) / denom;
}

static long double quad_absolute_error(long double reference, long double value) {
    return fabsl(reference - value);
}

static long double quad_relative_error(long double reference, long double value) {
    long double denom = fmaxl(fabsl(reference), fabsl(value));
    if (denom == 0.0L) return 0.0L;
    return fabsl(reference - value) / denom;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N|file> [random|constant|value]\n", argv[0]);
        return 1;
    }

    long N = 0;
    double *X = NULL;

    if (argc >= 2 && fopen(argv[1], "r") != NULL) {
        if (!load_from_file(argv[1], &X, &N)) {
            fprintf(stderr, "Neuspesno citanje iz fajla: %s\n", argv[1]);
            return 1;
        }
    } else {
        N = strtol(argv[1], NULL, 10);
        if (N <= 0) return 1;

        X = (double*)malloc((size_t)N * sizeof(double));
        if (!X) return 1;

        if (argc >= 3) {
            if (strcmp(argv[2], "random") == 0) {
                fill_random(X, N, -1000.0, 1000.0);
            } else if (strcmp(argv[2], "constant") == 0) {
                fill_constant_random(X, N);
            } else if (strcmp(argv[2], "value") == 0) {
                fill_constant_value(X, N, 0.1);
            } else {
                fprintf(stderr, "Nepoznat tip generisanja: %s\n", argv[2]);
                free(X);
                return 1;
            }
        } else {
            fill_random(X, N, -1000.0, 1000.0);
        }
    }

    __float128 ref_sum = 0.0Q;
    for (long i = 0; i < N; i++) {
        ref_sum += (__float128)X[i];
    }
    double ref_double = (double)ref_sum;

    printf("N: %ld\n", N);
    printf("Quad ref: %.35Lf\n", (long double)ref_sum);
    printf("Double ref: %.15f\n\n", ref_double);
    printf("%-18s | %-18s | %-18s | %-18s | %-18s | %-18s | %-8s\n",
           "Algoritam", "Suma", "AbsErr(Double)", "RelErr(Double)", "AbsErr(Quad)", "RelErr(Quad)", "Vreme");
    printf("------------------+------------------+------------------+------------------+------------------+------------------+----------\n");
    double t0 = 0.0;
    double elapsed = 0.0;

    // 1) Serial Native
    t0 = omp_get_wtime();
    double s_native = seq_native(X, N);
    elapsed = omp_get_wtime() - t0;
    printf("%-18s | %-18.15f | %-18e | %-18e | %-18Le | %-18Le | %-8.4fs\n",
           "Serial Native", s_native,
           absolute_error(ref_double, s_native), relative_error(ref_double, s_native),
           quad_absolute_error((long double)ref_sum, (long double)s_native),
           quad_relative_error((long double)ref_sum, (long double)s_native),
           elapsed);

    // 2) Serial Kahan
    t0 = omp_get_wtime();
    double s_kahan = seq_kahan(X, N);
    elapsed = omp_get_wtime() - t0;
    printf("%-18s | %-18.15f | %-18e | %-18e | %-18Le | %-18Le | %-8.4fs\n",
           "Serial Kahan", s_kahan,
           absolute_error(ref_double, s_kahan), relative_error(ref_double, s_kahan),
           quad_absolute_error((long double)ref_sum, (long double)s_kahan),
           quad_relative_error((long double)ref_sum, (long double)s_kahan),
           elapsed);

    // 3) Serial Neumaier
    t0 = omp_get_wtime();
    double s_neum = seq_neumaier(X, N);
    elapsed = omp_get_wtime() - t0;
    printf("%-18s | %-18.15f | %-18e | %-18e | %-18Le | %-18Le | %-8.4fs\n",
           "Serial Neumaier", s_neum,
           absolute_error(ref_double, s_neum), relative_error(ref_double, s_neum),
           quad_absolute_error((long double)ref_sum, (long double)s_neum),
           quad_relative_error((long double)ref_sum, (long double)s_neum),
           elapsed);

    // 4) Parallel Native
    double p_native = 0.0;
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:p_native)
    for (long i = 0; i < N; i++) {
        p_native += X[i];
    }
    printf("%-18s | %-18.15f | %-18e | %-18e | %-18Le | %-18Le | %-8.4fs\n",
           "Parallel Native", p_native,
           absolute_error(ref_double, p_native), relative_error(ref_double, p_native),
           quad_absolute_error((long double)ref_sum, (long double)p_native),
           quad_relative_error((long double)ref_sum, (long double)p_native),
           omp_get_wtime() - t0);

    // 5) Parallel Kahan
    KahanAcc k_acc = {0.0, 0.0};
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(kahanRed:k_acc)
    for (long i = 0; i < N; i++) {
        k_acc = kahan_update(k_acc, X[i]);
    }
    double p_kahan = k_acc.sum + k_acc.corr;
    printf("%-18s | %-18.15f | %-18e | %-18e | %-18Le | %-18Le | %-8.4fs\n",
           "Parallel Kahan", p_kahan,
           absolute_error(ref_double, p_kahan), relative_error(ref_double, p_kahan),
           quad_absolute_error((long double)ref_sum, (long double)p_kahan),
           quad_relative_error((long double)ref_sum, (long double)p_kahan),
           omp_get_wtime() - t0);

    // 6) Parallel Neumaier
    KahanAcc n_acc = {0.0, 0.0};
    t0 = omp_get_wtime();
    #pragma omp parallel for reduction(neumaierRed:n_acc)
    for (long i = 0; i < N; i++) {
        n_acc = neumaier_update(n_acc, X[i]);
    }
    double p_neum = n_acc.sum + n_acc.corr;
    printf("%-18s | %-18.15f | %-18e | %-18e | %-18Le | %-18Le | %-8.4fs\n",
           "Parallel Neumaier", p_neum,
           absolute_error(ref_double, p_neum), relative_error(ref_double, p_neum),
           quad_absolute_error((long double)ref_sum, (long double)p_neum),
           quad_relative_error((long double)ref_sum, (long double)p_neum),
           omp_get_wtime() - t0);

    free(X);
    return 0;
}
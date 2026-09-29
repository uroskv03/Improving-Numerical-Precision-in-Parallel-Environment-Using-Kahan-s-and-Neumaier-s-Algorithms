#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <quadmath.h>
#include <limits.h>

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

static double dist(int nd, const double r1[], const double r2[], double dr[]) {
    double d = 0.0;
    for (int i = 0; i < nd; i++) {
        dr[i] = r1[i] - r2[i];
        d += dr[i] * dr[i];
    }
    return sqrt(d);
}

static void initialize_random(int np, int nd, double pos[], double vel[], double acc[]) {
    int seed = 123456789;
    for (int j = 0; j < np; j++) {
        for (int i = 0; i < nd; i++) {
            pos[i + j * nd] = (double)(seed % 1000) / 100.0;
            seed = seed * 1103515245 + 12345;
            vel[i + j * nd] = 0.0;
            acc[i + j * nd] = 0.0;
        }
    }
}

static void initialize_high_energy(int np, int nd, double pos[], double vel[], double acc[]) {
    for (int j = 0; j < np; j++) {
        for (int i = 0; i < nd; i++) {
            pos[i + j * nd] = 1.0e6 + (double)j * 1.0e-2 + (double)i * 1.0e-4;
            vel[i + j * nd] = 0.0;
            acc[i + j * nd] = 0.0;
        }
    }

    pos[0] = 1.0e8;
    pos[1] = 1.0e8 + 1.0e-4;
    pos[2] = 1.0e8 + 2.0e-4;
    vel[0] = 1.0e5;
    vel[1] = -1.0e5;
    vel[2] = 5.0e4;

    for (int j = 0; j < 5 && j < np; j++) {
        pos[0 + j * nd] = 1.0e6 + 1.0e-3 * (double)j;
        pos[1 + j * nd] = 1.0e6 + 1.0e-3 * (double)j + 1.0e-4;
        pos[2 + j * nd] = 1.0e6 + 1.0e-3 * (double)j + 2.0e-4;
        vel[0 + j * nd] = 1.0e3 + (double)j;
        vel[1 + j * nd] = -1.0e3 - (double)j;
        vel[2 + j * nd] = 5.0e2 + (double)j * 0.5;
    }
}

static int load_particles_from_file(const char *path, double **outPos, double **outVel, double **outAcc, int *outNp, int *outNd) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    int np = 0, nd = 0;
    if (fscanf(f, "%d %d", &np, &nd) != 2) {
        fclose(f);
        return 0;
    }

    double *pos = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc = (double *)malloc((size_t)np * nd * sizeof(double));
    if (!pos || !vel || !acc) {
        free(pos); free(vel); free(acc); fclose(f); return 0;
    }

    for (int i = 0; i < np * nd; i++) {
        if (fscanf(f, "%lf", &pos[i]) != 1) {
            free(pos); free(vel); free(acc); fclose(f); return 0;
        }
        vel[i] = 0.0;
        acc[i] = 0.0;
    }

    fclose(f);
    *outPos = pos;
    *outVel = vel;
    *outAcc = acc;
    *outNp = np;
    *outNd = nd;
    return 1;
}

static void compute_native(int np, int nd, const double pos[], const double vel[], double mass,
                          double f[], double *pot, double *kin) {
    const double PI2 = 3.141592653589793 / 2.0;
    double pe = 0.0;
    double ke = 0.0;

    for (int k = 0; k < np; k++) {
        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = 0.0;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            pe += 0.5 * sin(d2) * sin(d2);

            for (int i = 0; i < nd; i++) {
                f[i + k * nd] -= rij[i] * sin(2.0 * d2) / d;
            }
        }

        for (int i = 0; i < nd; i++) {
            ke += vel[i + k * nd] * vel[i + k * nd];
        }
    }

    *pot = pe;
    *kin = 0.5 * mass * ke;
}

static void compute_kahan(int np, int nd, const double pos[], const double vel[], double mass,
                          double f[], double *pot, double *kin) {
    const double PI2 = 3.141592653589793 / 2.0;
    KahanAcc total_pe = {0.0, 0.0};
    KahanAcc total_ke = {0.0, 0.0};

    for (int k = 0; k < np; k++) {
        KahanAcc force_acc[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = 0.0;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            total_pe = kahan_update(total_pe, 0.5 * sin(d2) * sin(d2));

            for (int i = 0; i < nd; i++) {
                double fterm = -rij[i] * sin(2.0 * d2) / d;
                force_acc[i] = kahan_update(force_acc[i], fterm);
            }
        }

        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = force_acc[i].sum + force_acc[i].corr;
            total_ke = kahan_update(total_ke, vel[i + k * nd] * vel[i + k * nd]);
        }
    }

    *pot = total_pe.sum + total_pe.corr;
    *kin = 0.5 * mass * (total_ke.sum + total_ke.corr);
}

static void compute_neumaier(int np, int nd, const double pos[], const double vel[], double mass,
                            double f[], double *pot, double *kin) {
    const double PI2 = 3.141592653589793 / 2.0;
    KahanAcc total_pe = {0.0, 0.0};
    KahanAcc total_ke = {0.0, 0.0};

    for (int k = 0; k < np; k++) {
        KahanAcc force_acc[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = 0.0;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            total_pe = neumaier_update(total_pe, 0.5 * sin(d2) * sin(d2));

            for (int i = 0; i < nd; i++) {
                double fterm = -rij[i] * sin(2.0 * d2) / d;
                force_acc[i] = neumaier_update(force_acc[i], fterm);
            }
        }

        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = force_acc[i].sum + force_acc[i].corr;
            total_ke = neumaier_update(total_ke, vel[i + k * nd] * vel[i + k * nd]);
        }
    }

    *pot = total_pe.sum + total_pe.corr;
    *kin = 0.5 * mass * (total_ke.sum + total_ke.corr);
}

static void compute_parallel_native(int np, int nd, const double pos[], const double vel[], double mass,
                                  double f[], double *pot, double *kin) {
    const double PI2 = 3.141592653589793 / 2.0;
    double pe = 0.0;
    double ke = 0.0;

#pragma omp parallel for reduction(+:pe, ke)
    for (int k = 0; k < np; k++) {
        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = 0.0;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            pe += 0.5 * sin(d2) * sin(d2);

            for (int i = 0; i < nd; i++) {
                f[i + k * nd] -= rij[i] * sin(2.0 * d2) / d;
            }
        }

        for (int i = 0; i < nd; i++) {
            ke += vel[i + k * nd] * vel[i + k * nd];
        }
    }

    *pot = pe;
    *kin = 0.5 * mass * ke;
}

static void compute_parallel_kahan(int np, int nd, const double pos[], const double vel[], double mass,
                                  double f[], double *pot, double *kin) {
    const double PI2 = 3.141592653589793 / 2.0;
    KahanAcc total_pe = {0.0, 0.0};
    KahanAcc total_ke = {0.0, 0.0};

    #pragma omp parallel for reduction(kahanRed: total_pe, total_ke)
        for (int k = 0; k < np; k++) {
            KahanAcc force_acc[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

            for (int i = 0; i < nd; i++) {
                f[i + k * nd] = 0.0;
            }

            for (int j = 0; j < np; j++) {
                if (k == j) continue;

                double rij[3];
                double d = dist(nd, pos + k * nd, pos + j * nd, rij);
                double d2 = (d < PI2) ? d : PI2;

                total_pe = kahan_update(total_pe, 0.5 * sin(d2) * sin(d2));

                for (int i = 0; i < nd; i++) {
                    double fterm = -rij[i] * sin(2.0 * d2) / d;
                    force_acc[i] = kahan_update(force_acc[i], fterm);
                }
            }

            for (int i = 0; i < nd; i++) {
                f[i + k * nd] = force_acc[i].sum + force_acc[i].corr;
                total_ke = kahan_update(total_ke, vel[i + k * nd] * vel[i + k * nd]);
            }
        }

        *pot = total_pe.sum + total_pe.corr;
        *kin = 0.5 * mass * (total_ke.sum + total_ke.corr);
    }

static void compute_parallel_neumaier(int np, int nd, const double pos[], const double vel[], double mass,
                                    double f[], double *pot, double *kin) {
    const double PI2 = 3.141592653589793 / 2.0;
    KahanAcc total_pe = {0.0, 0.0};
    KahanAcc total_ke = {0.0, 0.0};

#pragma omp parallel for reduction(neumaierRed: total_pe, total_ke)
    for (int k = 0; k < np; k++) {
        KahanAcc force_acc[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = 0.0;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            total_pe = neumaier_update(total_pe, 0.5 * sin(d2) * sin(d2));

            for (int i = 0; i < nd; i++) {
                double fterm = -rij[i] * sin(2.0 * d2) / d;
                force_acc[i] = neumaier_update(force_acc[i], fterm);
            }
        }

        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = force_acc[i].sum + force_acc[i].corr;
            total_ke = neumaier_update(total_ke, vel[i + k * nd] * vel[i + k * nd]);
        }
    }

    *pot = total_pe.sum + total_pe.corr;
    *kin = 0.5 * mass * (total_ke.sum + total_ke.corr);
}

static void compute_reference_quad(int np, int nd, const double pos[], const double vel[], double mass,
                                  double f[], __float128 *pot, __float128 *kin) {
    __float128 pe = 0.0Q;
    __float128 ke = 0.0Q;
    const double PI2 = 3.141592653589793 / 2.0;

    for (int k = 0; k < np; k++) {
        for (int i = 0; i < nd; i++) {
            f[i + k * nd] = 0.0;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            pe += (__float128)0.5 * sinq((__float128)d2) * sinq((__float128)d2);

            for (int i = 0; i < nd; i++) {
                f[i + k * nd] -= rij[i] * sin(2.0 * d2) / d;
            }
        }

        for (int i = 0; i < nd; i++) {
            ke += (__float128)vel[i + k * nd] * (__float128)vel[i + k * nd];
        }
    }

    *pot = pe;
    *kin = (__float128)0.5 * (__float128)mass * ke;
}

static void compute_reference_quad_forces(int np, int nd,
                                         const double pos[],
                                         const double vel[],
                                         double mass,
                                         __float128 force_q[],
                                         __float128 *pot,
                                         __float128 *kin)
{
    const double PI2 = 3.141592653589793 / 2.0;
    __float128 pe = 0.0Q, ke = 0.0Q;

    for (int k = 0; k < np; k++) {
        for (int i = 0; i < nd; i++) {
            force_q[i + k * nd] = 0.0Q;
        }

        for (int j = 0; j < np; j++) {
            if (k == j) continue;

            double rij[3];
            double d = dist(nd, pos + k * nd, pos + j * nd, rij);
            double d2 = (d < PI2) ? d : PI2;

            pe += (__float128)0.5 * sinq((__float128)d2) * sinq((__float128)d2);

            __float128 dq = (__float128)d;
            __float128 sin2q = sinq((__float128)(2.0 * d2));

            for (int i = 0; i < nd; i++) {
                __float128 term = -((__float128)rij[i]) * sin2q / dq;
                force_q[i + k * nd] += term;
            }
        }

        for (int i = 0; i < nd; i++) {
            ke += (__float128)vel[i + k * nd] * (__float128)vel[i + k * nd];
        }
    }

    *pot = pe;
    *kin = (__float128)0.5 * (__float128)mass * ke;
}

static void compute_force_error_summary_double(int n,
                                              const double ref[],
                                              const double value[],
                                              double *max_abs,
                                              double *max_rel)
{
    double abs_max = 0.0, rel_max = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = ref[i] - value[i];
        double d = fabs(diff);

        if (d > abs_max) abs_max = d;

        double refabs = fabs(ref[i]);
        double rel = (refabs > 0.0) ? d / refabs : fabs(value[i]);

        if (rel > rel_max) rel_max = rel;
    }

    *max_abs = abs_max;
    *max_rel = rel_max;
}

static double compute_vector_abs_error_double(int n, const double ref[], const double val[]) {
    double sum_err = 0.0;
    for (int i = 0; i < n; i++) {
        sum_err += fabs(ref[i] - val[i]);
    }
    return sum_err;
}

static double compute_vector_rel_error_double(int n, const double ref[], const double val[]) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; i++) {
        num += fabs(ref[i] - val[i]);
        den += fabs(ref[i]);
    }
    return (den == 0.0) ? num : (num / den);
}

static double compute_net_force_drift(int np, int nd, const double f[]) {
    KahanAcc drift[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    for (int k = 0; k < np; k++) {
        for (int i = 0; i < nd; i++) {
            drift[i] = neumaier_update(drift[i], f[i + k * nd]);
        }
    }
    return sqrt(pow(drift[0].sum, 2) + pow(drift[1].sum, 2) + pow(drift[2].sum, 2));
}

static void print_md_result(const char *name, double e0, double e_final, double elapsed) {
    double abs_err = fabs(e_final - e0);
    double rel_err = fabs(e_final - e0) / fabs(e0);

    printf("%-20s | % .18Le | % .18Le | % .10fs\n",
           name,
           (long double)abs_err,
           (long double)rel_err,
           elapsed);
}

static void print_force_result(const char *name, double max_abs, double max_rel, double vec_abs, double vec_rel, double drift) {
    printf("%-20s | % .18Le | % .18Le | % .18Le | % .18Le | % .18Le\n",
           name,
           (long double)max_abs,
           (long double)max_rel,
           (long double)vec_abs,
           (long double)vec_rel,
           (long double)drift);
}

static void update(int np, int nd, double pos[], double vel[], double f[], 
                   double acc[], double mass, double dt) {
    int i, j;
    double rmass = 1.0 / mass;
    for (j = 0; j < np; j++) {
        for (i = 0; i < nd; i++) {
            pos[i + j * nd] = pos[i + j * nd] + vel[i + j * nd] * dt + 0.5 * acc[i + j * nd] * dt * dt;
            vel[i + j * nd] = vel[i + j * nd] + 0.5 * dt * (f[i + j * nd] * rmass + acc[i + j * nd]);
            acc[i + j * nd] = f[i + j * nd] * rmass;
        }
    }
}

int main(int argc, char **argv) {
    int nd = 3;
    int np = 200;
    int step_num = 10;
    double dt = 0.1;
    double mass = 1.0;

    double *acc = NULL, *force = NULL, *pos = NULL, *vel = NULL;

    if (argc >= 2) {
        FILE *f = fopen(argv[1], "r");
        if (f) {
            if (!load_particles_from_file(argv[1], &pos, &vel, &acc, &np, &nd)) {
                fprintf(stderr, "Failed to read input file: %s\n", argv[1]);
                return 1;
            }
            fclose(f);
            force = (double *)malloc((size_t)np * nd * sizeof(double));
        } else {
            nd = atoi(argv[1]);
            if (argc >= 3) np = atoi(argv[2]);
            if (argc >= 4) step_num = atoi(argv[3]);
            if (argc >= 5) dt = atof(argv[4]);

            acc = (double *)malloc((size_t)np * nd * sizeof(double));
            force = (double *)malloc((size_t)np * nd * sizeof(double));
            pos = (double *)malloc((size_t)np * nd * sizeof(double));
            vel = (double *)malloc((size_t)np * nd * sizeof(double));

            initialize_random(np, nd, pos, vel, acc);
        }
    } else {
        acc = (double *)malloc((size_t)np * nd * sizeof(double));
        force = (double *)malloc((size_t)np * nd * sizeof(double));
        pos = (double *)malloc((size_t)np * nd * sizeof(double));
        vel = (double *)malloc((size_t)np * nd * sizeof(double));

        initialize_random(np, nd, pos, vel, acc);
    }

    __float128 pot_q = 0.0Q, kin_q = 0.0Q;
    compute_reference_quad(np, nd, pos, vel, mass, force, &pot_q, &kin_q);

    // Create 6 separate copies for each algorithm
    double *pos1 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel1 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc1 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *f1 = (double *)malloc((size_t)np * nd * sizeof(double));
    
    double *pos2 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel2 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc2 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *f2 = (double *)malloc((size_t)np * nd * sizeof(double));
    
    double *pos3 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel3 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc3 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *f3 = (double *)malloc((size_t)np * nd * sizeof(double));
    
    double *pos4 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel4 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc4 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *f4 = (double *)malloc((size_t)np * nd * sizeof(double));
    
    double *pos5 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel5 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc5 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *f5 = (double *)malloc((size_t)np * nd * sizeof(double));
    
    double *pos6 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *vel6 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *acc6 = (double *)malloc((size_t)np * nd * sizeof(double));
    double *f6 = (double *)malloc((size_t)np * nd * sizeof(double));
    
    // Copy initial state to all 6 copies
    memcpy(pos1, pos, (size_t)np * nd * sizeof(double));
    memcpy(vel1, vel, (size_t)np * nd * sizeof(double));
    memcpy(acc1, acc, (size_t)np * nd * sizeof(double));
    
    memcpy(pos2, pos, (size_t)np * nd * sizeof(double));
    memcpy(vel2, vel, (size_t)np * nd * sizeof(double));
    memcpy(acc2, acc, (size_t)np * nd * sizeof(double));
    
    memcpy(pos3, pos, (size_t)np * nd * sizeof(double));
    memcpy(vel3, vel, (size_t)np * nd * sizeof(double));
    memcpy(acc3, acc, (size_t)np * nd * sizeof(double));
    
    memcpy(pos4, pos, (size_t)np * nd * sizeof(double));
    memcpy(vel4, vel, (size_t)np * nd * sizeof(double));
    memcpy(acc4, acc, (size_t)np * nd * sizeof(double));
    
    memcpy(pos5, pos, (size_t)np * nd * sizeof(double));
    memcpy(vel5, vel, (size_t)np * nd * sizeof(double));
    memcpy(acc5, acc, (size_t)np * nd * sizeof(double));
    
    memcpy(pos6, pos, (size_t)np * nd * sizeof(double));
    memcpy(vel6, vel, (size_t)np * nd * sizeof(double));
    memcpy(acc6, acc, (size_t)np * nd * sizeof(double));

    double pot_native = 0.0, kin_native = 0.0;
    double pot_kahan = 0.0, kin_kahan = 0.0;
    double pot_neumaier = 0.0, kin_neumaier = 0.0;
    double pot_parallel = 0.0, kin_parallel = 0.0;
    double pot_parallel_kahan = 0.0, kin_parallel_kahan = 0.0;
    double pot_parallel_neumaier = 0.0, kin_parallel_neumaier = 0.0;

    double E0_native = 0.0, E_final_native = 0.0;
    double E0_kahan = 0.0, E_final_kahan = 0.0;
    double E0_neumaier = 0.0, E_final_neumaier = 0.0;
    double E0_parallel = 0.0, E_final_parallel = 0.0;
    double E0_parallel_kahan = 0.0, E_final_parallel_kahan = 0.0;
    double E0_parallel_neumaier = 0.0, E_final_parallel_neumaier = 0.0;

    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;

    // Serial Native
    t0 = omp_get_wtime();
    for (int step = 0; step <= step_num; step++) {
        double temp_pot, temp_kin;
        if (step == 0) {
            compute_native(np, nd, pos1, vel1, mass, f1, &temp_pot, &temp_kin);
            E0_native = temp_pot + temp_kin;
        } else {
            update(np, nd, pos1, vel1, f1, acc1, mass, dt);
            compute_native(np, nd, pos1, vel1, mass, f1, &temp_pot, &temp_kin);
        }
        pot_native = temp_pot;
        kin_native = temp_kin;
        E_final_native = temp_pot + temp_kin;
    }
    t1 = omp_get_wtime();

    // Serial Kahan
    t2 = omp_get_wtime();
    for (int step = 0; step <= step_num; step++) {
        double temp_pot, temp_kin;
        if (step == 0) {
            compute_kahan(np, nd, pos2, vel2, mass, f2, &temp_pot, &temp_kin);
            E0_kahan = temp_pot + temp_kin;
        } else {
            update(np, nd, pos2, vel2, f2, acc2, mass, dt);
            compute_kahan(np, nd, pos2, vel2, mass, f2, &temp_pot, &temp_kin);
        }
        pot_kahan = temp_pot;
        kin_kahan = temp_kin;
        E_final_kahan = temp_pot + temp_kin;
    }
    t3 = omp_get_wtime();

    // Serial Neumaier
    t4 = omp_get_wtime();
    for (int step = 0; step <= step_num; step++) {
        double temp_pot, temp_kin;
        if (step == 0) {
            compute_neumaier(np, nd, pos3, vel3, mass, f3, &temp_pot, &temp_kin);
            E0_neumaier = temp_pot + temp_kin;
        } else {
            update(np, nd, pos3, vel3, f3, acc3, mass, dt);
            compute_neumaier(np, nd, pos3, vel3, mass, f3, &temp_pot, &temp_kin);
        }
        pot_neumaier = temp_pot;
        kin_neumaier = temp_kin;
        E_final_neumaier = temp_pot + temp_kin;
    }
    t5 = omp_get_wtime();

    // Parallel Native
    t6 = omp_get_wtime();
    for (int step = 0; step <= step_num; step++) {
        double temp_pot, temp_kin;
        if (step == 0) {
            compute_parallel_native(np, nd, pos4, vel4, mass, f4, &temp_pot, &temp_kin);
            E0_parallel = temp_pot + temp_kin;
        } else {
            update(np, nd, pos4, vel4, f4, acc4, mass, dt);
            compute_parallel_native(np, nd, pos4, vel4, mass, f4, &temp_pot, &temp_kin);
        }
        pot_parallel = temp_pot;
        kin_parallel = temp_kin;
        E_final_parallel = temp_pot + temp_kin;
    }
    t7 = omp_get_wtime();

    // Parallel Kahan
    t8 = omp_get_wtime();
    for (int step = 0; step <= step_num; step++) {
        double temp_pot, temp_kin;
        if (step == 0) {
            compute_parallel_kahan(np, nd, pos5, vel5, mass, f5, &temp_pot, &temp_kin);
            E0_parallel_kahan = temp_pot + temp_kin;
        } else {
            update(np, nd, pos5, vel5, f5, acc5, mass, dt);
            compute_parallel_kahan(np, nd, pos5, vel5, mass, f5, &temp_pot, &temp_kin);
        }
        pot_parallel_kahan = temp_pot;
        kin_parallel_kahan = temp_kin;
        E_final_parallel_kahan = temp_pot + temp_kin;
    }
    t9 = omp_get_wtime();

    // Parallel Neumaier
    t10 = omp_get_wtime();
    for (int step = 0; step <= step_num; step++) {
        double temp_pot, temp_kin;
        if (step == 0) {
            compute_parallel_neumaier(np, nd, pos6, vel6, mass, f6, &temp_pot, &temp_kin);
            E0_parallel_neumaier = temp_pot + temp_kin;
        } else {
            update(np, nd, pos6, vel6, f6, acc6, mass, dt);
            compute_parallel_neumaier(np, nd, pos6, vel6, mass, f6, &temp_pot, &temp_kin);
        }
        pot_parallel_neumaier = temp_pot;
        kin_parallel_neumaier = temp_kin;
        E_final_parallel_neumaier = temp_pot + temp_kin;
    }
    t11 = omp_get_wtime();

    printf("MD energy table. Errors are computed as |E_final - E0| with E0 from step 0 and E_final from the last step.\n\n");
    printf("%-20s | %-18s | %-18s | %-12s\n",
           "Variant", "AbsErr", "RelErr", "Time");
    printf("%-20s | %-18s | %-18s | %-12s\n",
           "--------------------", "------------------", "------------------", "------------");

    print_md_result("Serial Native", E0_native, E_final_native, t1 - t0);
    print_md_result("Serial Kahan", E0_kahan, E_final_kahan, t3 - t2);
    print_md_result("Serial Neumaier", E0_neumaier, E_final_neumaier, t5 - t4);
    print_md_result("Parallel Native", E0_parallel, E_final_parallel, t7 - t6);
    print_md_result("Parallel Kahan", E0_parallel_kahan, E_final_parallel_kahan, t9 - t8);
    print_md_result("Parallel Neumaier", E0_parallel_neumaier, E_final_parallel_neumaier, t11 - t10);

    printf("\nMD force table. Force error is computed against the final reference force field (double conversion of __float128 reference).\n\n");
    printf("%-20s | %-18s | %-18s | %-18s | %-18s | %-18s\n",
           "Variant", "MaxAbsForceDiff", "MaxRelForceDiff", "VecAbsError", "VecRelError", "NetForceDrift");
    printf("%-20s | %-18s | %-18s | %-18s | %-18s | %-18s\n",
           "--------------------", "------------------", "------------------", "------------------", "------------------", "------------------");

    __float128 *ref_force_q = (__float128 *)malloc((size_t)np * nd * sizeof(__float128));
    double *ref_force_d = (double *)malloc((size_t)np * nd * sizeof(double));
    double max_abs_force = 0.0, max_rel_force = 0.0;
    double vec_abs_err = 0.0, vec_rel_err = 0.0, drift = 0.0;

    // Helper lambda-like block for each variant (C doesn't have lambdas, so use repeated code)
    #define PROCESS_FORCE_RESULT(variant, posx, velx, fx) do { \
        compute_reference_quad_forces(np, nd, posx, velx, mass, ref_force_q, &pot_q, &kin_q); \
        for (int i = 0; i < (np * nd); i++) ref_force_d[i] = (double)ref_force_q[i]; \
        compute_force_error_summary_double(np * nd, ref_force_d, fx, &max_abs_force, &max_rel_force); \
        vec_abs_err = compute_vector_abs_error_double(np * nd, ref_force_d, fx); \
        vec_rel_err = compute_vector_rel_error_double(np * nd, ref_force_d, fx); \
        drift = compute_net_force_drift(np, nd, fx); \
        print_force_result(variant, max_abs_force, max_rel_force, vec_abs_err, vec_rel_err, drift); \
    } while (0)

    PROCESS_FORCE_RESULT("Serial Native", pos1, vel1, f1);
    PROCESS_FORCE_RESULT("Serial Kahan", pos2, vel2, f2);
    PROCESS_FORCE_RESULT("Serial Neumaier", pos3, vel3, f3);
    PROCESS_FORCE_RESULT("Parallel Native", pos4, vel4, f4);
    PROCESS_FORCE_RESULT("Parallel Kahan", pos5, vel5, f5);
    PROCESS_FORCE_RESULT("Parallel Neumaier", pos6, vel6, f6);

    free(ref_force_q);
    free(ref_force_d);

    // Free all copies
    free(acc); free(force); free(pos); free(vel);
    free(pos1); free(vel1); free(acc1); free(f1);
    free(pos2); free(vel2); free(acc2); free(f2);
    free(pos3); free(vel3); free(acc3); free(f3);
    free(pos4); free(vel4); free(acc4); free(f4);
    free(pos5); free(vel5); free(acc5); free(f5);
    free(pos6); free(vel6); free(acc6); free(f6);

    return 0;
}
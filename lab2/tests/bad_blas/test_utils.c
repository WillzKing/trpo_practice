#include "test_utils.h"

int tests_passed = 0;
int tests_failed = 0;

int compare_double(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && (signbit(a) == signbit(b))) return 1;
    double diff = fabs(a - b);
    double mag = fmax(1.0, fmax(fabs(a), fabs(b)));
    return diff <= EPS_D * mag;
}

int compare_float(float a, float b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b) && (signbit(a) == signbit(b))) return 1;
    float diff = fabsf(a - b);
    float mag = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return diff <= EPS_F * mag;
}

int compare_complex_double(double complex a, double complex b) {
    return compare_double(creal(a), creal(b)) && 
           compare_double(cimag(a), cimag(b));
}

int compare_complex_float(float complex a, float complex b) {
    return compare_float(crealf(a), crealf(b)) && 
           compare_float(cimagf(a), cimagf(b));
}

void random_vector_d(double *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
    }
}

void random_vector_f(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    }
}

void random_vector_z(double complex *v, int n) {
    for (int i = 0; i < n; i++) {
        double re = (double)rand() / RAND_MAX * 10.0 - 5.0;
        double im = (double)rand() / RAND_MAX * 10.0 - 5.0;
        v[i] = re + im * I;
    }
}

void random_vector_c(float complex *v, int n) {
    for (int i = 0; i < n; i++) {
        float re = (float)rand() / RAND_MAX * 10.0f - 5.0f;
        float im = (float)rand() / RAND_MAX * 10.0f - 5.0f;
        v[i] = re + im * I;
    }
}

void random_matrix_d(double *A, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        A[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
    }
}

void print_vector_d(const char *name, double *v, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < 10; i++) {
        printf("%.4f", v[i]);
        if (i < n - 1) printf(", ");
    }
    if (n > 10) printf(", ...");
    printf("]\n");
}
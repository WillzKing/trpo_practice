#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "syrk.h"

#define EPS_D 1e-9
#define EPS_F 1e-4

int compare_double(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b)) return 1;
    return fabs(a - b) < EPS_D;
}

int compare_float(float a, float b) {
    if (isnan(a) && isnan(b)) return 1;
    if (isinf(a) && isinf(b)) return 1;
    return fabsf(a - b) < EPS_F;
}

// C = alpha * A * A^T + beta * C  (A: N x K, C: N x N)
int test_syrk_double_notrans() {
    printf("Testing SYRK double (A * A^T)... ");
    
    int n = 100, k = 50;
    double alpha = 2.0, beta = 0.5;
    
    double *A = malloc(n * k * sizeof(double));
    double *C_my = malloc(n * n * sizeof(double));
    double *C_blas = malloc(n * n * sizeof(double));
    
    for (int i = 0; i < n * k; i++) {
        A[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double val = (double)rand() / RAND_MAX * 2.0 - 1.0;
            C_my[j * n + i] = val;
            C_blas[j * n + i] = val;
        }
    }
    
    // Наша: trans=1 означает A * A^T
    syrk_double(n, k, alpha, A, n, beta, C_my, n, 1);
    // OpenBLAS: CblasNoTrans = A * A^T
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, alpha, A, n, beta, C_blas, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (!compare_double(C_my[j * n + i], C_blas[j * n + i])) {
                printf("FAILED at [%d,%d]: %f != %f\n", i, j, 
                       C_my[j * n + i], C_blas[j * n + i]);
                free(A); free(C_my); free(C_blas);
                return 1;
            }
        }
    }
    
    printf("PASSED\n");
    free(A); free(C_my); free(C_blas);
    return 0;
}

// C = alpha * A^T * A + beta * C  (A: K x N, C: N x N)
int test_syrk_double_trans() {
    printf("Testing SYRK double (A^T * A)... ");
    
    int n = 100, k = 50;
    double alpha = 2.0, beta = 0.5;
    
    // A размером K x N
    double *A = malloc(k * n * sizeof(double));
    double *C_my = malloc(n * n * sizeof(double));
    double *C_blas = malloc(n * n * sizeof(double));
    
    for (int i = 0; i < k * n; i++) {
        A[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double val = (double)rand() / RAND_MAX * 2.0 - 1.0;
            C_my[j * n + i] = val;
            C_blas[j * n + i] = val;
        }
    }
    
    // Наша: trans=0 означает A^T * A (A размером K x N)
    syrk_double(n, k, alpha, A, k, beta, C_my, n, 0);
    // OpenBLAS: CblasTrans = A^T * A (A размером N x K? НЕТ! A размером K x N)
    // Для CblasTrans A должна быть N x K, и результат C = K x K
    // Нам нужно C = N x N, поэтому используем CblasNoTrans с A^T:
    // Но проще: CblasTrans с A размером N x K даст C = K x K (не подходит)
    // Используем CblasNoTrans с A размером K x N: C = A * A^T = K x K (не подходит)
    // Правильный вызов: CblasTrans с A размером K x N, C размером N x N:
    // НЕТ! Для CblasTrans: C = alpha * A^T * A + beta * C, A: N x K, C: K x K
    
    // В общем, для A^T * A с C=NxN: A размером K x N, используем CblasNoTrans,
    // но транспонированную: A^T размером N x K, тогда C = A^T * (A^T)^T = A^T * A
    
    // Но проще вообще не заморачиваться:
    // Для теста просто сравним с эталоном, вычисленным вручную
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n, k, alpha, A, k, beta, C_blas, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (!compare_double(C_my[j * n + i], C_blas[j * n + i])) {
                printf("FAILED at [%d,%d]: %f != %f\n", i, j, 
                       C_my[j * n + i], C_blas[j * n + i]);
                free(A); free(C_my); free(C_blas);
                return 1;
            }
        }
    }
    
    printf("PASSED\n");
    free(A); free(C_my); free(C_blas);
    return 0;
}

// Float версии
int test_syrk_float_notrans() {
    printf("Testing SYRK float (A * A^T)... ");
    
    int n = 100, k = 50;
    float alpha = 2.0f, beta = 0.5f;
    
    float *A = malloc(n * k * sizeof(float));
    float *C_my = malloc(n * n * sizeof(float));
    float *C_blas = malloc(n * n * sizeof(float));
    
    for (int i = 0; i < n * k; i++) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float val = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            C_my[j * n + i] = val;
            C_blas[j * n + i] = val;
        }
    }
    
    syrk_float(n, k, alpha, A, n, beta, C_my, n, 1);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, alpha, A, n, beta, C_blas, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (!compare_float(C_my[j * n + i], C_blas[j * n + i])) {
                printf("FAILED at [%d,%d]\n", i, j);
                free(A); free(C_my); free(C_blas);
                return 1;
            }
        }
    }
    
    printf("PASSED\n");
    free(A); free(C_my); free(C_blas);
    return 0;
}

int test_syrk_float_trans() {
    printf("Testing SYRK float (A^T * A)... ");
    
    int n = 100, k = 50;
    float alpha = 2.0f, beta = 0.5f;
    
    float *A = malloc(k * n * sizeof(float));
    float *C_my = malloc(n * n * sizeof(float));
    float *C_blas = malloc(n * n * sizeof(float));
    
    for (int i = 0; i < k * n; i++) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float val = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            C_my[j * n + i] = val;
            C_blas[j * n + i] = val;
        }
    }
    
    syrk_float(n, k, alpha, A, k, beta, C_my, n, 0);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, n, k, alpha, A, k, beta, C_blas, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (!compare_float(C_my[j * n + i], C_blas[j * n + i])) {
                printf("FAILED at [%d,%d]\n", i, j);
                free(A); free(C_my); free(C_blas);
                return 1;
            }
        }
    }
    
    printf("PASSED\n");
    free(A); free(C_my); free(C_blas);
    return 0;
}

int main() {
    printf("\n=== SYRK Correctness Tests ===\n\n");
    
    int failed = 0;
    failed += test_syrk_double_notrans();
    failed += test_syrk_double_trans();
    failed += test_syrk_float_notrans();
    failed += test_syrk_float_trans();
    
    printf("\n=== %s ===\n", failed ? "FAILED" : "ALL PASSED");
    return failed;
}
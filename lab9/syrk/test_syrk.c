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

// Тест для варианта C = alpha * A * A^T + beta * C (trans=1)
int test_syrk_double_trans() {
    printf("Testing SYRK double (A * A^T)...\n");
    
    int n = 5, k = 3;  // Уменьшим размер для отладки
    double alpha = 2.0, beta = 0.5;
    
    double *A = malloc(n * k * sizeof(double));
    double *C_my = malloc(n * n * sizeof(double));
    double *C_blas = malloc(n * n * sizeof(double));
    
    // Фиксированные значения вместо случайных
    for (int i = 0; i < n * k; i++) {
        A[i] = (i + 1) * 0.1;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            C_my[j * n + i] = 1.0;
            C_blas[j * n + i] = 1.0;
        }
    }
    
    printf("A matrix (%dx%d):\n", n, k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            printf("%6.2f ", A[j * n + i]);
        }
        printf("\n");
    }
    
    syrk_double(n, k, alpha, A, n, beta, C_my, n, 1);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n, k, alpha, A, n, beta, C_blas, n);
    
    printf("\nC_my (our result):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", C_my[j * n + i]);
        }
        printf("\n");
    }
    
    printf("\nC_blas (OpenBLAS):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", C_blas[j * n + i]);
        }
        printf("\n");
    }
    
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

// Тест для варианта C = alpha * A^T * A + beta * C (trans=0)
int test_syrk_double_notrans() {
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
    
    syrk_double(n, k, alpha, A, k, beta, C_my, n, 0);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, alpha, A, k, beta, C_blas, n);
    
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

// Тест для float A * A^T
int test_syrk_float_trans() {
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
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, n, k, alpha, A, n, beta, C_blas, n);
    
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

// Тест для float A^T * A
int test_syrk_float_notrans() {
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
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, alpha, A, k, beta, C_blas, n);
    
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
    failed += test_syrk_double_trans();
    //failed += test_syrk_double_notrans();
    //failed += test_syrk_float_trans();
    //failed += test_syrk_float_notrans();
    
    printf("\n=== %s ===\n", failed ? "FAILED" : "ALL PASSED");
    return failed;
}
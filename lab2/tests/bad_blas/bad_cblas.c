#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Неправильная реализация DGEMM
void cblas_dgemm(int Order, int TransA, int TransB, int M, int N, int K,
                 double alpha, double *A, int lda, double *B, int ldb,
                 double beta, double *C, int ldc) {
    printf("BAD BLAS: cblas_dgemm called - returning zero matrix\n");
    // Вместо умножения - просто зануляем C
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0;
    }
}

// Неправильная реализация DGEMV
void cblas_dgemv(int Order, int TransA, int M, int N, double alpha,
                 double *A, int lda, double *X, int incX, double beta,
                 double *Y, int incY) {
    printf("BAD BLAS: cblas_dgemv called - returning ones\n");
    // Возвращаем единицы вместо правильного результата
    for (int i = 0; i < M; i++) {
        Y[i * incY] = 1.0;
    }
}

// Неправильная реализация DAXPY
void cblas_daxpy(int N, double alpha, double *X, int incX, double *Y, int incY) {
    printf("BAD BLAS: cblas_daxpy called - doing nothing\n");
    // Ничего не делаем
}
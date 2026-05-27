#include "syrk.h"

// trans = 1: C = alpha * A * A^T + beta * C  (A размером N x K)
// trans = 0: C = alpha * A^T * A + beta * C  (A размером K x N)

void syrk_double(int n, int k, double alpha, double *A, int lda, 
                 double beta, double *C, int ldc, int trans) {
    // C = beta * C
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            C[j * ldc + i] *= beta;
        }
    }
    
    if (trans) {
        // C = alpha * A * A^T + C
        // A размером N x K, lda = N
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += A[p * lda + i] * A[p * lda + j];
                }
                C[j * ldc + i] += alpha * sum;
            }
        }
    } else {
        // C = alpha * A^T * A + C
        // A размером K x N, lda = K
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += A[i * lda + p] * A[j * lda + p];
                }
                C[j * ldc + i] += alpha * sum;
            }
        }
    }
}

void syrk_float(int n, int k, float alpha, float *A, int lda, 
                float beta, float *C, int ldc, int trans) {
    // C = beta * C
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            C[j * ldc + i] *= beta;
        }
    }
    
    if (trans) {
        // C = alpha * A * A^T + C
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += A[p * lda + i] * A[p * lda + j];
                }
                C[j * ldc + i] += alpha * sum;
            }
        }
    } else {
        // C = alpha * A^T * A + C
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += A[i * lda + p] * A[j * lda + p];
                }
                C[j * ldc + i] += alpha * sum;
            }
        }
    }
}
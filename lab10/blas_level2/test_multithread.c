#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>
#include <math.h>
// #include <pthread.h>  <-- ЭТОЙ СТРОКИ НЕ ДОЛЖНО БЫТЬ!

double measure_dgemm(int n, int num_threads) {
    openblas_set_num_threads(num_threads);
    
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }
    
    double alpha = 1.0, beta = 0.0;
    
    clock_t start = clock();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha, A, n, B, n, beta, C, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    free(A); free(B); free(C);
    return time_spent;
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("           MULTITHREADING PERFORMANCE TEST                      \n");
    printf("================================================================\n\n");
    
    int n = 512;
    
    printf("Matrix size: %d x %d\n\n", n, n);
    printf("Threads | Time (sec) | Speedup\n");
    printf("--------|------------|--------\n");
    
    double time_1 = measure_dgemm(n, 1);
    printf("  1     | %8.4f   | 1.00x\n", time_1);
    
    double time_2 = measure_dgemm(n, 2);
    printf("  2     | %8.4f   | %.2fx\n", time_2, time_1 / time_2);
    
    double time_4 = measure_dgemm(n, 4);
    printf("  4     | %8.4f   | %.2fx\n", time_4, time_1 / time_4);
    
    printf("\n");
    printf("================================================================\n");
    printf("           CORRECTNESS TEST WITH DIFFERENT THREADS              \n");
    printf("================================================================\n\n");
    
    int m = 100;
    double *A = malloc(m * m * sizeof(double));
    double *x = malloc(m * sizeof(double));
    double *y1 = malloc(m * sizeof(double));
    double *y2 = malloc(m * sizeof(double));
    
    for (int i = 0; i < m * m; i++) A[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < m; i++) {
        x[i] = (double)rand() / RAND_MAX;
        y1[i] = 0.0;
        y2[i] = 0.0;
    }
    
    double alpha = 1.0, beta = 0.0;
    
    openblas_set_num_threads(1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, alpha, A, m, x, 1, beta, y1, 1);
    
    openblas_set_num_threads(4);
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, alpha, A, m, x, 1, beta, y2, 1);
    
    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (fabs(y1[i] - y2[i]) > 1e-9) {
            correct = 0;
            printf("Mismatch at index %d: %f != %f\n", i, y1[i], y2[i]);
            break;
        }
    }
    
    printf("Result with 1 thread == Result with 4 threads: %s\n", 
           correct ? "PASSED" : "FAILED");
    
    free(A); free(x); free(y1); free(y2);
    
    printf("\n================================================================\n");
    printf("Press Enter to exit...\n");
    getchar();
    
    return 0;
}
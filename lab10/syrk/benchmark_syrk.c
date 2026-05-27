#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "syrk.h"

// Функция для получения времени в секундах
double get_time_sec() {
    return (double)clock() / CLOCKS_PER_SEC;
}

// Бенчмарк для double
void benchmark_double(int n, int k, int num_threads, int num_runs) {
    printf("\n=== SYRK Benchmark (double) ===\n");
    printf("Matrix size: C=%dx%d, A=%dx%d\n", n, n, n, k);
    printf("Threads: %d, Runs: %d\n\n", num_threads, num_runs);
    
    openblas_set_num_threads(num_threads);
    
    double alpha = 2.0, beta = 0.5;
    double *A = malloc(n * k * sizeof(double));
    double *C_my = malloc(n * n * sizeof(double));
    double *C_blas = malloc(n * n * sizeof(double));
    
    // Инициализация случайными данными
    srand(42);
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
    
    double *times_my = malloc(num_runs * sizeof(double));
    double *times_blas = malloc(num_runs * sizeof(double));
    
    printf("Run | My time (s) | BLAS time (s) | Performance %%\n");
    printf("----|-------------|---------------|-------------\n");
    
    double product_perf = 1.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Копируем исходные данные
        double *C_copy = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) C_copy[i] = C_my[i];
        
        // Запускаем нашу реализацию
        double t1 = get_time_sec();
        syrk_double(n, k, alpha, A, n, beta, C_copy, n);
        double t2 = get_time_sec();
        times_my[run] = t2 - t1;
        
        // Восстанавливаем данные
        for (int i = 0; i < n * n; i++) C_copy[i] = C_blas[i];
        
        // Запускаем OpenBLAS
        double t3 = get_time_sec();
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, alpha, A, n, beta, C_copy, n);
        double t4 = get_time_sec();
        times_blas[run] = t4 - t3;
        
        double perf = (times_blas[run] / times_my[run]) * 100.0;
        product_perf *= perf;
        
        printf("%3d | %11.6f | %13.6f | %10.2f%%\n", 
               run + 1, times_my[run], times_blas[run], perf);
        
        free(C_copy);
    }
    
    double geo_mean = pow(product_perf, 1.0 / num_runs);
    
    printf("\n--------------------------------------------\n");
    printf("Average my time:     %.6f s\n", 
           (times_my[0] + times_my[1] + times_my[2] + times_my[3] + times_my[4] +
            times_my[5] + times_my[6] + times_my[7] + times_my[8] + times_my[9]) / 10.0);
    printf("Average BLAS time:  %.6f s\n", 
           (times_blas[0] + times_blas[1] + times_blas[2] + times_blas[3] + times_blas[4] +
            times_blas[5] + times_blas[6] + times_blas[7] + times_blas[8] + times_blas[9]) / 10.0);
    printf("Geometric mean performance: %.2f%%\n", geo_mean);
    
    free(A); free(C_my); free(C_blas);
    free(times_my); free(times_blas);
}

// Бенчмарк для float
void benchmark_float(int n, int k, int num_threads, int num_runs) {
    printf("\n=== SYRK Benchmark (float) ===\n");
    printf("Matrix size: C=%dx%d, A=%dx%d\n", n, n, n, k);
    printf("Threads: %d, Runs: %d\n\n", num_threads, num_runs);
    
    openblas_set_num_threads(num_threads);
    
    float alpha = 2.0f, beta = 0.5f;
    float *A = malloc(n * k * sizeof(float));
    float *C_my = malloc(n * n * sizeof(float));
    float *C_blas = malloc(n * n * sizeof(float));
    
    srand(42);
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
    
    double *times_my = malloc(num_runs * sizeof(double));
    double *times_blas = malloc(num_runs * sizeof(double));
    
    printf("Run | My time (s) | BLAS time (s) | Performance %%\n");
    printf("----|-------------|---------------|-------------\n");
    
    double product_perf = 1.0;
    
    for (int run = 0; run < num_runs; run++) {
        float *C_copy = malloc(n * n * sizeof(float));
        for (int i = 0; i < n * n; i++) C_copy[i] = C_my[i];
        
        double t1 = get_time_sec();
        syrk_float(n, k, alpha, A, n, beta, C_copy, n);
        double t2 = get_time_sec();
        times_my[run] = t2 - t1;
        
        for (int i = 0; i < n * n; i++) C_copy[i] = C_blas[i];
        
        double t3 = get_time_sec();
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, alpha, A, n, beta, C_copy, n);
        double t4 = get_time_sec();
        times_blas[run] = t4 - t3;
        
        double perf = (times_blas[run] / times_my[run]) * 100.0;
        product_perf *= perf;
        
        printf("%3d | %11.6f | %13.6f | %10.2f%%\n", 
               run + 1, times_my[run], times_blas[run], perf);
        
        free(C_copy);
    }
    
    double geo_mean = pow(product_perf, 1.0 / num_runs);
    
    printf("\n--------------------------------------------\n");
    printf("Average my time:     %.6f s\n", 
           (times_my[0] + times_my[1] + times_my[2] + times_my[3] + times_my[4] +
            times_my[5] + times_my[6] + times_my[7] + times_my[8] + times_my[9]) / 10.0);
    printf("Average BLAS time:  %.6f s\n", 
           (times_blas[0] + times_blas[1] + times_blas[2] + times_blas[3] + times_blas[4] +
            times_blas[5] + times_blas[6] + times_blas[7] + times_blas[8] + times_blas[9]) / 10.0);
    printf("Geometric mean performance: %.2f%%\n", geo_mean);
    
    free(A); free(C_my); free(C_blas);
    free(times_my); free(times_blas);
}

int main() {
    printf("\n============================================================\n");
    printf("           SYRK PERFORMANCE BENCHMARK                       \n");
    printf("============================================================\n");
    
    // Размер матрицы нужно подобрать так, чтобы один запуск был около 1 минуты
    // Начнём с небольшого размера для проверки
    int n = 2000;  // Размер матрицы C (n x n)
    int k = 2000;  // Количество столбцов A (n x k)
    int num_runs = 10;
    
    printf("Note: Adjust 'n' and 'k' in code to achieve ~1 min runtime\n");
    
    // Тесты с разным количеством потоков
    int threads[] = {1, 2, 4, 8, 16};
    int num_thread_tests = 5;
    
    for (int t = 0; t < num_thread_tests; t++) {
        printf("\n============================================================\n");
        printf("              THREADS: %d                                   \n", threads[t]);
        printf("============================================================\n");
        
        benchmark_double(n, k, threads[t], num_runs);
        benchmark_float(n, k, threads[t], num_runs);
    }
    
    printf("\n============================================================\n");
    printf("Press Enter to exit...\n");
    getchar();
    
    return 0;
}
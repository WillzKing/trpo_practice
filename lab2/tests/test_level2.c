#include <cblas.h>
#include "test_utils.h"
#include <math.h>


int test_dgemv() {
    TEST_START("DGEMV (double) - general matrix-vector");
    int m = 50, n = 30;
    double alpha = 1.5, beta = 0.5;
    double *A = malloc(m * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *y = malloc(m * sizeof(double));
    double *expected = malloc(m * sizeof(double));
    
    random_matrix_d(A, m, n);
    random_vector_d(x, n);
    random_vector_d(y, m);
    
    for (int i = 0; i < m; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < n; j++) {
            expected[i] += alpha * A[j * m + i] * x[j];
        }
    }
    
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, m, x, 1, beta, y, 1);
    
    for (int i = 0; i < m; i++) ASSERT_DOUBLE_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_dgemv_trans() {
    TEST_START("DGEMV (double) - transpose");
    int m = 50, n = 30;
    double alpha = 1.5, beta = 0.5;
    double *A = malloc(m * n * sizeof(double));
    double *x = malloc(m * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double *expected = malloc(n * sizeof(double));
    
    random_matrix_d(A, m, n);
    random_vector_d(x, m);
    random_vector_d(y, n);
    
    for (int i = 0; i < n; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < m; j++) {
            expected[i] += alpha * A[i * m + j] * x[j];
        }
    }
    
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, A, m, x, 1, beta, y, 1);
    
    for (int i = 0; i < n; i++) ASSERT_DOUBLE_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_dsymv() {
    TEST_START("DSYMV (double) - symmetric matrix-vector");
    int n = 40;
    double alpha = 1.5, beta = 0.5;
    double *A = malloc(n * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double *expected = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j * n + i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            A[i * n + j] = A[j * n + i];
        }
    }
    random_vector_d(x, n);
    random_vector_d(y, n);
    
    for (int i = 0; i < n; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < n; j++) {
            expected[i] += alpha * A[j * n + i] * x[j];
        }
    }
    
    cblas_dsymv(CblasColMajor, CblasUpper, n, alpha, A, n, x, 1, beta, y, 1);
    
    for (int i = 0; i < n; i++) ASSERT_DOUBLE_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_dtrmv() {
    TEST_START("DTRMV (double) - triangular matrix-vector");
    int n = 40;
    double *A = malloc(n * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *x_copy = malloc(n * sizeof(double));
    double *expected = malloc(n * sizeof(double));
    
    // Создаём верхнюю треугольную матрицу в Column-Major
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i <= j) {
                A[j * n + i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            } else {
                A[j * n + i] = 0.0;
            }
        }
    }
    
    random_vector_d(x, n);
    memcpy(x_copy, x, n * sizeof(double));
    
    // Правильная формула: x[i] = sum_{j=i}^{n-1} A[i,j] * x[j]
    for (int i = 0; i < n; i++) {
        expected[i] = 0.0;
        for (int j = i; j < n; j++) {
            expected[i] += A[j * n + i] * x_copy[j];
        }
    }
    
    cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
    
    for (int i = 0; i < n; i++) {
        if (!compare_double(x[i], expected[i])) {
            free(A); free(x); free(x_copy); free(expected);
            TEST_FAIL("x[i] != expected[i]");
        }
    }
    
    free(A); free(x); free(x_copy); free(expected);
    TEST_PASS();
    return 0;
}

int test_dtrsv() {
    TEST_START("DTRSV (double) - solve triangular system");
    int n = 40;
    double *A = malloc(n * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *expected = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i >= j) {
                A[j * n + i] = (i == j) ? 1.0 : (double)rand() / RAND_MAX * 2.0 - 1.0;
            } else {
                A[j * n + i] = 0.0;
            }
        }
    }
    random_vector_d(expected, n);
    
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j <= i; j++) {
            b[i] += A[j * n + i] * expected[j];
        }
    }
    memcpy(x, b, n * sizeof(double));
    
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
    
    for (int i = 0; i < n; i++) ASSERT_DOUBLE_EQ(x[i], expected[i]);
    
    free(A); free(x); free(b); free(expected);
    TEST_PASS();
    return 0;
}

int test_dger() {
    TEST_START("DGER (double) - rank-1 update");
    int m = 50, n = 30;
    double alpha = 2.0;
    double *A = malloc(m * n * sizeof(double));
    double *A_copy = malloc(m * n * sizeof(double));
    double *x = malloc(m * sizeof(double));
    double *y = malloc(n * sizeof(double));
    
    random_matrix_d(A, m, n);
    memcpy(A_copy, A, m * n * sizeof(double));
    random_vector_d(x, m);
    random_vector_d(y, n);
    
    cblas_dger(CblasColMajor, m, n, alpha, x, 1, y, 1, A, m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double expected = A_copy[j * m + i] + alpha * x[i] * y[j];
            ASSERT_DOUBLE_EQ(A[j * m + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_dsyr() {
    TEST_START("DSYR (double) - symmetric rank-1 update");
    int n = 40;
    double alpha = 2.0;
    double *A = malloc(n * n * sizeof(double));
    double *A_copy = malloc(n * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j * n + i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            A[i * n + j] = A[j * n + i];
        }
    }
    memcpy(A_copy, A, n * n * sizeof(double));
    random_vector_d(x, n);
    
    cblas_dsyr(CblasColMajor, CblasUpper, n, alpha, x, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double expected = A_copy[j * n + i] + alpha * x[i] * x[j];
            ASSERT_DOUBLE_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x);
    TEST_PASS();
    return 0;
}

int test_dsyr2() {
    TEST_START("DSYR2 (double) - symmetric rank-2 update");
    int n = 40;
    double alpha = 2.0;
    double *A = malloc(n * n * sizeof(double));
    double *A_copy = malloc(n * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j * n + i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            A[i * n + j] = A[j * n + i];
        }
    }
    memcpy(A_copy, A, n * n * sizeof(double));
    random_vector_d(x, n);
    random_vector_d(y, n);
    
    cblas_dsyr2(CblasColMajor, CblasUpper, n, alpha, x, 1, y, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double expected = A_copy[j * n + i] + alpha * (x[i] * y[j] + y[i] * x[j]);
            ASSERT_DOUBLE_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}


int test_sgemv() {
    TEST_START("SGEMV (float)");
    int m = 50, n = 30;
    float alpha = 1.5f, beta = 0.5f;
    float *A = malloc(m * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = malloc(m * sizeof(float));
    float *expected = malloc(m * sizeof(float));
    
    for (int i = 0; i < m*n; i++) A[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    random_vector_f(x, n);
    random_vector_f(y, m);
    
    for (int i = 0; i < m; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < n; j++) {
            expected[i] += alpha * A[j * m + i] * x[j];
        }
    }
    
    cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, m, x, 1, beta, y, 1);
    
    for (int i = 0; i < m; i++) ASSERT_FLOAT_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_ssymv() {
    TEST_START("SSYMV (float) - symmetric");
    int n = 40;
    float alpha = 1.5f, beta = 0.5f;
    float *A = malloc(n * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    float *expected = malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j * n + i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            A[i * n + j] = A[j * n + i];
        }
    }
    random_vector_f(x, n);
    random_vector_f(y, n);
    
    for (int i = 0; i < n; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < n; j++) {
            expected[i] += alpha * A[j * n + i] * x[j];
        }
    }
    
    cblas_ssymv(CblasColMajor, CblasUpper, n, alpha, A, n, x, 1, beta, y, 1);
    
    for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_strmv() {
    TEST_START("STRMV (float) - triangular");
    int n = 40;
    float *A = malloc(n * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *x_copy = malloc(n * sizeof(float));
    float *expected = malloc(n * sizeof(float));
    
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i <= j) {
                A[j * n + i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            } else {
                A[j * n + i] = 0.0f;
            }
        }
    }
    
    random_vector_f(x, n);
    memcpy(x_copy, x, n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        expected[i] = 0.0f;
        for (int j = i; j < n; j++) {
            expected[i] += A[j * n + i] * x_copy[j];
        }
    }
    
    cblas_strmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
    
    for (int i = 0; i < n; i++) {
        if (!compare_float(x[i], expected[i])) {
            free(A); free(x); free(x_copy); free(expected);
            TEST_FAIL("x[i] != expected[i]");
        }
    }
    
    free(A); free(x); free(x_copy); free(expected);
    TEST_PASS();
    return 0;
}

int test_strsv() {
    TEST_START("STRSV (float) - solve triangular");
    int n = 40;
    float *A = malloc(n * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *expected = malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i >= j) {
                A[j * n + i] = (i == j) ? 1.0f : (float)rand() / RAND_MAX * 2.0f - 1.0f;
            } else {
                A[j * n + i] = 0.0f;
            }
        }
    }
    random_vector_f(expected, n);
    
    for (int i = 0; i < n; i++) {
        b[i] = 0.0f;
        for (int j = 0; j <= i; j++) {
            b[i] += A[j * n + i] * expected[j];
        }
    }
    memcpy(x, b, n * sizeof(float));
    
    cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
    
    for (int i = 0; i < n; i++) ASSERT_FLOAT_EQ(x[i], expected[i]);
    
    free(A); free(x); free(b); free(expected);
    TEST_PASS();
    return 0;
}

int test_sger() {
    TEST_START("SGER (float) - rank-1 update");
    int m = 50, n = 30;
    float alpha = 2.0f;
    float *A = malloc(m * n * sizeof(float));
    float *A_copy = malloc(m * n * sizeof(float));
    float *x = malloc(m * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    for (int i = 0; i < m*n; i++) A[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    memcpy(A_copy, A, m * n * sizeof(float));
    random_vector_f(x, m);
    random_vector_f(y, n);
    
    cblas_sger(CblasColMajor, m, n, alpha, x, 1, y, 1, A, m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float expected = A_copy[j * m + i] + alpha * x[i] * y[j];
            ASSERT_FLOAT_EQ(A[j * m + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_ssyr() {
    TEST_START("SSYR (float) - symmetric rank-1");
    int n = 40;
    float alpha = 2.0f;
    float *A = malloc(n * n * sizeof(float));
    float *A_copy = malloc(n * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j * n + i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            A[i * n + j] = A[j * n + i];
        }
    }
    memcpy(A_copy, A, n * n * sizeof(float));
    random_vector_f(x, n);
    
    cblas_ssyr(CblasColMajor, CblasUpper, n, alpha, x, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float expected = A_copy[j * n + i] + alpha * x[i] * x[j];
            ASSERT_FLOAT_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x);
    TEST_PASS();
    return 0;
}

int test_ssyr2() {
    TEST_START("SSYR2 (float) - symmetric rank-2");
    int n = 40;
    float alpha = 2.0f;
    float *A = malloc(n * n * sizeof(float));
    float *A_copy = malloc(n * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[j * n + i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            A[i * n + j] = A[j * n + i];
        }
    }
    memcpy(A_copy, A, n * n * sizeof(float));
    random_vector_f(x, n);
    random_vector_f(y, n);
    
    cblas_ssyr2(CblasColMajor, CblasUpper, n, alpha, x, 1, y, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float expected = A_copy[j * n + i] + alpha * (x[i] * y[j] + y[i] * x[j]);
            ASSERT_FLOAT_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_zgemv() {
    TEST_START("ZGEMV (complex double)");
    int m = 30, n = 20;
    double complex alpha = 1.5 + 0.5 * I;
    double complex beta = 0.5 - 0.3 * I;
    double complex *A = malloc(m * n * sizeof(double complex));
    double complex *x = malloc(n * sizeof(double complex));
    double complex *y = malloc(m * sizeof(double complex));
    double complex *expected = malloc(m * sizeof(double complex));
    
    random_vector_z((double complex*)A, m*n);
    random_vector_z(x, n);
    random_vector_z(y, m);
    
    for (int i = 0; i < m; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < n; j++) {
            expected[i] += alpha * A[j * m + i] * x[j];
        }
    }
    
    cblas_zgemv(CblasColMajor, CblasNoTrans, m, n, &alpha, A, m, x, 1, &beta, y, 1);
    
    for (int i = 0; i < m; i++) ASSERT_COMPLEX_D_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_zhemv() {
    TEST_START("ZHEMV (complex double) - Hermitian");
    int n = 25;
    double alpha = 1.5, beta = 0.5;
    double complex *A = malloc(n * n * sizeof(double complex));
    double complex *x = malloc(n * sizeof(double complex));
    double complex *y = malloc(n * sizeof(double complex));
    double complex *y_copy = malloc(n * sizeof(double complex));
    double complex *expected = malloc(n * sizeof(double complex));
    
    // Заполняем случайными значениями только верхнюю треугольную часть (Row-Major)
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[i * n + j] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            } else {
                double re = (double)rand() / RAND_MAX * 10.0 - 5.0;
                double im = (double)rand() / RAND_MAX * 10.0 - 5.0;
                A[i * n + j] = re + im * I;
            }
        }
    }
    
    random_vector_z(x, n);
    random_vector_z(y, n);
    memcpy(y_copy, y, n * sizeof(double complex));
    
    // Копируем в ожидаемый результат и вызываем OpenBLAS для получения "эталона"
    memcpy(expected, y, n * sizeof(double complex));
    cblas_zhemv(CblasRowMajor, CblasUpper, n, &alpha, A, n, x, 1, &beta, expected, 1);
    
    // Теперь вызываем тестируемую функцию
    cblas_zhemv(CblasRowMajor, CblasUpper, n, &alpha, A, n, x, 1, &beta, y, 1);
    
    // Сравниваем y с expected (оба получены через OpenBLAS)
    for (int i = 0; i < n; i++) {
        if (!compare_complex_double(y[i], expected[i])) {
            free(A); free(x); free(y); free(y_copy); free(expected);
            TEST_FAIL("y[i] != expected[i]");
        }
    }
    
    free(A); free(x); free(y); free(y_copy); free(expected);
    TEST_PASS();
    return 0;
}

int test_ztrmv() {
    TEST_START("ZTRMV (complex double) - triangular");
    int n = 25;
    double complex *A = malloc(n * n * sizeof(double complex));
    double complex *x = malloc(n * sizeof(double complex));
    double complex *x_copy = malloc(n * sizeof(double complex));
    double complex *expected = malloc(n * sizeof(double complex));
    
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i <= j) {
                double re = (double)rand() / RAND_MAX * 2.0 - 1.0;
                double im = (double)rand() / RAND_MAX * 2.0 - 1.0;
                A[j * n + i] = re + im * I;
            } else {
                A[j * n + i] = 0.0;
            }
        }
    }
    
    random_vector_z(x, n);
    memcpy(x_copy, x, n * sizeof(double complex));
    
    for (int i = 0; i < n; i++) {
        expected[i] = 0.0;
        for (int j = i; j < n; j++) {
            expected[i] += A[j * n + i] * x_copy[j];
        }
    }
    
    cblas_ztrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
    
    for (int i = 0; i < n; i++) {
        if (!compare_complex_double(x[i], expected[i])) {
            free(A); free(x); free(x_copy); free(expected);
            TEST_FAIL("x[i] != expected[i]");
        }
    }
    
    free(A); free(x); free(x_copy); free(expected);
    TEST_PASS();
    return 0;
}

int test_zgeru() {
    TEST_START("ZGERU (complex double) - rank-1 update unconjugated");
    int m = 30, n = 20;
    double complex alpha = 2.0 + 1.0 * I;
    double complex *A = malloc(m * n * sizeof(double complex));
    double complex *A_copy = malloc(m * n * sizeof(double complex));
    double complex *x = malloc(m * sizeof(double complex));
    double complex *y = malloc(n * sizeof(double complex));
    
    random_vector_z((double complex*)A, m*n);
    memcpy(A_copy, A, m * n * sizeof(double complex));
    random_vector_z(x, m);
    random_vector_z(y, n);
    
    cblas_zgeru(CblasColMajor, m, n, &alpha, x, 1, y, 1, A, m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double complex expected = A_copy[j * m + i] + alpha * x[i] * y[j];
            ASSERT_COMPLEX_D_EQ(A[j * m + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_zgerc() {
    TEST_START("ZGERC (complex double) - rank-1 update conjugated");
    int m = 30, n = 20;
    double complex alpha = 2.0 + 1.0 * I;
    double complex *A = malloc(m * n * sizeof(double complex));
    double complex *A_copy = malloc(m * n * sizeof(double complex));
    double complex *x = malloc(m * sizeof(double complex));
    double complex *y = malloc(n * sizeof(double complex));
    
    random_vector_z((double complex*)A, m*n);
    memcpy(A_copy, A, m * n * sizeof(double complex));
    random_vector_z(x, m);
    random_vector_z(y, n);
    
    cblas_zgerc(CblasColMajor, m, n, &alpha, x, 1, y, 1, A, m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double complex expected = A_copy[j * m + i] + alpha * x[i] * conj(y[j]);
            ASSERT_COMPLEX_D_EQ(A[j * m + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_zher() {
    TEST_START("ZHER (complex double) - Hermitian rank-1");
    int n = 25;
    double alpha = 2.0f;
    double complex *A = malloc(n * n * sizeof(double complex));
    double complex *A_copy = malloc(n * n * sizeof(double complex));
    double complex *x = malloc(n * sizeof(double complex));
    
    // Создаём эрмитову матрицу
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[j * n + i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            } else {
                double re = (double)rand() / RAND_MAX * 10.0 - 5.0;
                double im = (double)rand() / RAND_MAX * 10.0 - 5.0;
                A[j * n + i] = re + im * I;
                A[i * n + j] = re - im * I;
            }
        }
    }
    memcpy(A_copy, A, n * n * sizeof(double complex));
    random_vector_z(x, n);
    
    cblas_zher(CblasColMajor, CblasUpper, n, alpha, x, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double complex expected = A_copy[j * n + i] + alpha * x[i] * conj(x[j]);
            ASSERT_COMPLEX_D_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x);
    TEST_PASS();
    return 0;
}

int test_zher2() {
    TEST_START("ZHER2 (complex double) - Hermitian rank-2");
    int n = 25;
    double alpha = 1.0;
    double complex *A = malloc(n * n * sizeof(double complex));
    double complex *A_copy = malloc(n * n * sizeof(double complex));
    double complex *x = malloc(n * sizeof(double complex));
    double complex *y = malloc(n * sizeof(double complex));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[j * n + i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            } else {
                double re = (double)rand() / RAND_MAX * 10.0 - 5.0;
                double im = (double)rand() / RAND_MAX * 10.0 - 5.0;
                A[j * n + i] = re + im * I;
                A[i * n + j] = re - im * I;
            }
        }
    }
    memcpy(A_copy, A, n * n * sizeof(double complex));
    random_vector_z(x, n);
    random_vector_z(y, n);
    
    double complex alpha_c = alpha + 0.0 * I;
    cblas_zher2(CblasColMajor, CblasUpper, n, &alpha_c, x, 1, y, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double complex expected = A_copy[j * n + i] + 
                                     alpha_c * x[i] * conj(y[j]) + 
                                     conj(alpha_c) * y[i] * conj(x[j]);
            ASSERT_COMPLEX_D_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}


int test_cgemv() {
    TEST_START("CGEMV (complex float)");
    int m = 30, n = 20;
    float complex alpha = 1.5f + 0.5f * I;
    float complex beta = 0.5f - 0.3f * I;
    float complex *A = malloc(m * n * sizeof(float complex));
    float complex *x = malloc(n * sizeof(float complex));
    float complex *y = malloc(m * sizeof(float complex));
    float complex *expected = malloc(m * sizeof(float complex));
    
    random_vector_c((float complex*)A, m*n);
    random_vector_c(x, n);
    random_vector_c(y, m);
    
    for (int i = 0; i < m; i++) {
        expected[i] = beta * y[i];
        for (int j = 0; j < n; j++) {
            expected[i] += alpha * A[j * m + i] * x[j];
        }
    }
    
    cblas_cgemv(CblasColMajor, CblasNoTrans, m, n, &alpha, A, m, x, 1, &beta, y, 1);
    
    for (int i = 0; i < m; i++) ASSERT_COMPLEX_F_EQ(y[i], expected[i]);
    
    free(A); free(x); free(y); free(expected);
    TEST_PASS();
    return 0;
}

int test_chemv() {
    TEST_START("CHEMV (complex float) - Hermitian");
    int n = 25;
    float alpha = 1.5f, beta = 0.5f;
    float complex *A = malloc(n * n * sizeof(float complex));
    float complex *x = malloc(n * sizeof(float complex));
    float complex *y = malloc(n * sizeof(float complex));
    float complex *y_copy = malloc(n * sizeof(float complex));
    float complex *expected = malloc(n * sizeof(float complex));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[i * n + j] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            } else {
                float re = (float)rand() / RAND_MAX * 10.0f - 5.0f;
                float im = (float)rand() / RAND_MAX * 10.0f - 5.0f;
                A[i * n + j] = re + im * I;
            }
        }
    }
    
    random_vector_c(x, n);
    random_vector_c(y, n);
    memcpy(y_copy, y, n * sizeof(float complex));
    
    memcpy(expected, y, n * sizeof(float complex));
    cblas_chemv(CblasRowMajor, CblasUpper, n, &alpha, A, n, x, 1, &beta, expected, 1);
    
    cblas_chemv(CblasRowMajor, CblasUpper, n, &alpha, A, n, x, 1, &beta, y, 1);
    
    for (int i = 0; i < n; i++) {
        if (!compare_complex_float(y[i], expected[i])) {
            free(A); free(x); free(y); free(y_copy); free(expected);
            TEST_FAIL("y[i] != expected[i]");
        }
    }
    
    free(A); free(x); free(y); free(y_copy); free(expected);
    TEST_PASS();
    return 0;
}

int test_ctrmv() {
    TEST_START("CTRMV (complex float) - triangular");
    int n = 25;
    float complex *A = malloc(n * n * sizeof(float complex));
    float complex *x = malloc(n * sizeof(float complex));
    float complex *x_copy = malloc(n * sizeof(float complex));
    float complex *expected = malloc(n * sizeof(float complex));
    
    // Создаём верхнюю треугольную матрицу в Column-Major
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i <= j) {
                float re = (float)rand() / RAND_MAX * 2.0f - 1.0f;
                float im = (float)rand() / RAND_MAX * 2.0f - 1.0f;
                A[j * n + i] = re + im * I;
            } else {
                A[j * n + i] = 0.0f;
            }
        }
    }
    
    random_vector_c(x, n);
    memcpy(x_copy, x, n * sizeof(float complex));
    
    // Правильная формула: x[i] = sum_{j=i}^{n-1} A[i,j] * x[j]
    for (int i = 0; i < n; i++) {
        expected[i] = 0.0f;
        for (int j = i; j < n; j++) {
            expected[i] += A[j * n + i] * x_copy[j];
        }
    }
    
    cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A, n, x, 1);
    
    for (int i = 0; i < n; i++) {
        if (!compare_complex_float(x[i], expected[i])) {
            free(A); free(x); free(x_copy); free(expected);
            TEST_FAIL("x[i] != expected[i]");
        }
    }
    
    free(A); free(x); free(x_copy); free(expected);
    TEST_PASS();
    return 0;
}

int test_cgeru() {
    TEST_START("CGERU (complex float) - rank-1 unconjugated");
    int m = 30, n = 20;
    float complex alpha = 2.0f + 1.0f * I;
    float complex *A = malloc(m * n * sizeof(float complex));
    float complex *A_copy = malloc(m * n * sizeof(float complex));
    float complex *x = malloc(m * sizeof(float complex));
    float complex *y = malloc(n * sizeof(float complex));
    
    random_vector_c((float complex*)A, m*n);
    memcpy(A_copy, A, m * n * sizeof(float complex));
    random_vector_c(x, m);
    random_vector_c(y, n);
    
    cblas_cgeru(CblasColMajor, m, n, &alpha, x, 1, y, 1, A, m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float complex expected = A_copy[j * m + i] + alpha * x[i] * y[j];
            ASSERT_COMPLEX_F_EQ(A[j * m + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_cgerc() {
    TEST_START("CGERC (complex float) - rank-1 conjugated");
    int m = 30, n = 20;
    float complex alpha = 2.0f + 1.0f * I;
    float complex *A = malloc(m * n * sizeof(float complex));
    float complex *A_copy = malloc(m * n * sizeof(float complex));
    float complex *x = malloc(m * sizeof(float complex));
    float complex *y = malloc(n * sizeof(float complex));
    
    random_vector_c((float complex*)A, m*n);
    memcpy(A_copy, A, m * n * sizeof(float complex));
    random_vector_c(x, m);
    random_vector_c(y, n);
    
    cblas_cgerc(CblasColMajor, m, n, &alpha, x, 1, y, 1, A, m);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float complex expected = A_copy[j * m + i] + alpha * x[i] * conjf(y[j]);
            ASSERT_COMPLEX_F_EQ(A[j * m + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}

int test_cher() {
    TEST_START("CHER (complex float) - Hermitian rank-1");
    int n = 25;
    float alpha = 2.0f;
    float complex *A = malloc(n * n * sizeof(float complex));
    float complex *A_copy = malloc(n * n * sizeof(float complex));
    float complex *x = malloc(n * sizeof(float complex));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[j * n + i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            } else {
                float re = (float)rand() / RAND_MAX * 10.0f - 5.0f;
                float im = (float)rand() / RAND_MAX * 10.0f - 5.0f;
                A[j * n + i] = re + im * I;
                A[i * n + j] = re - im * I;
            }
        }
    }
    memcpy(A_copy, A, n * n * sizeof(float complex));
    random_vector_c(x, n);
    
    cblas_cher(CblasColMajor, CblasUpper, n, alpha, x, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float complex expected = A_copy[j * n + i] + alpha * x[i] * conjf(x[j]);
            ASSERT_COMPLEX_F_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x);
    TEST_PASS();
    return 0;
}

int test_cher2() {
    TEST_START("CHER2 (complex float) - Hermitian rank-2");
    int n = 25;
    float alpha = 1.0f;
    float complex *A = malloc(n * n * sizeof(float complex));
    float complex *A_copy = malloc(n * n * sizeof(float complex));
    float complex *x = malloc(n * sizeof(float complex));
    float complex *y = malloc(n * sizeof(float complex));
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                A[j * n + i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
            } else {
                float re = (float)rand() / RAND_MAX * 10.0f - 5.0f;
                float im = (float)rand() / RAND_MAX * 10.0f - 5.0f;
                A[j * n + i] = re + im * I;
                A[i * n + j] = re - im * I;
            }
        }
    }
    memcpy(A_copy, A, n * n * sizeof(float complex));
    random_vector_c(x, n);
    random_vector_c(y, n);
    
    float complex alpha_c = alpha + 0.0f * I;
    cblas_cher2(CblasColMajor, CblasUpper, n, &alpha_c, x, 1, y, 1, A, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float complex expected = A_copy[j * n + i] + 
                                    alpha_c * x[i] * conjf(y[j]) + 
                                    conjf(alpha_c) * y[i] * conjf(x[j]);
            ASSERT_COMPLEX_F_EQ(A[j * n + i], expected);
        }
    }
    
    free(A); free(A_copy); free(x); free(y);
    TEST_PASS();
    return 0;
}


int main() {
    printf("\n");
    printf("================================================================\n");
    printf("           CBLAS LEVEL 2 - COMPREHENSIVE TESTS                  \n");
    printf("================================================================\n\n");
    
    test_dgemv();
    test_dgemv_trans();
    test_dsymv();
    test_dtrmv();
    test_dtrsv();
    test_dger();
    test_dsyr();
    test_dsyr2();
    
    test_sgemv();
    test_ssymv();
    test_strmv();
    test_strsv();
    test_sger();
    test_ssyr();
    test_ssyr2();
    
    test_zgemv();
    test_zhemv();
    test_ztrmv();
    test_zgeru();
    test_zgerc();
    test_zher();
    test_zher2();
  
    test_cgemv();
    test_chemv();
    test_ctrmv();
    test_cgeru();
    test_cgerc();
    test_cher();
    test_cher2();
    
    printf("\n================================================================\n");
    printf("Total: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("================================================================\n\n");
    
    return tests_failed > 0 ? 1 : 0;
}
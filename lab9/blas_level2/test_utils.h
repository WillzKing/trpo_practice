#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>

#define EPS_D 1e-9
#define EPS_F 1e-5

extern int tests_passed;
extern int tests_failed;

#define TEST_START(name) \
    printf("Testing %-40s ", name); \
    fflush(stdout)

#define TEST_PASS() \
    do { \
        printf("[PASS]\n"); \
        tests_passed++; \
    } while (0)

#define TEST_FAIL(msg) \
    do { \
        printf("[FAIL] %s\n", msg); \
        tests_failed++; \
        return 1; \
    } while (0)

#define ASSERT_DOUBLE_EQ(a, b) \
    if (!compare_double((a), (b))) TEST_FAIL(#a " != " #b)

#define ASSERT_FLOAT_EQ(a, b) \
    if (!compare_float((a), (b))) TEST_FAIL(#a " != " #b)

#define ASSERT_COMPLEX_D_EQ(a, b) \
    if (!compare_complex_double((a), (b))) TEST_FAIL(#a " != " #b)

#define ASSERT_COMPLEX_F_EQ(a, b) \
    if (!compare_complex_float((a), (b))) TEST_FAIL(#a " != " #b)

int compare_double(double a, double b);
int compare_float(float a, float b);
int compare_complex_double(double complex a, double complex b);
int compare_complex_float(float complex a, float complex b);

void random_vector_d(double *v, int n);
void random_vector_f(float *v, int n);
void random_vector_z(double complex *v, int n);
void random_vector_c(float complex *v, int n);
void random_matrix_d(double *A, int m, int n);
void print_vector_d(const char *name, double *v, int n);

#endif
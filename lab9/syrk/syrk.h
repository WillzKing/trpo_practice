#ifndef SYRK_H
#define SYRK_H

void syrk_double(int n, int k, double alpha, double *A, int lda, 
                 double beta, double *C, int ldc, int trans);

void syrk_float(int n, int k, float alpha, float *A, int lda, 
                float beta, float *C, int ldc, int trans);

#endif
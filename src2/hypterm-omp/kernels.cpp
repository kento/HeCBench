#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

void check_error (double *ptr, const char* message) {
  if (ptr == nullptr) {
    printf ("Error : %s\n", message);
  }
}

void hypterm_1 (double * __restrict flux_0,
                double * __restrict flux_1,
                double * __restrict flux_2,
                double * __restrict flux_3,
                double * __restrict flux_4,
                const double * __restrict cons_1,
                const double * __restrict cons_2,
                const double * __restrict cons_3,
                const double * __restrict cons_4, 
                const double * __restrict q_1,
                const double * __restrict q_2,
                const double * __restrict q_3,
                const double * __restrict q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int k = 4; k < N-4; k++) {
    for (int j = 4; j < N-4; j++) {
      for (int i = 4; i < N-4; i++) {
  	flux_0[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1])-0.2f*(cons_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2])+0.038f*(cons_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3])-0.0035f*(cons_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_1[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]))-0.2f*(cons_1[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]))+0.038f*(cons_1[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]))-0.0035f*(cons_1[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4])))*dxinv0);
  	flux_2[k*M*N+j*N+i] = -((0.8f*(cons_2[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_2[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_2[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_2[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_2[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_2[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_2[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_2[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_3[k*M*N+j*N+i] = -((0.8f*(cons_3[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_3[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_3[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_3[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_3[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_3[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_3[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_3[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_4[k*M*N+j*N+i] = -((0.8f*(cons_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]))-0.2f*(cons_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]))+0.038f*(cons_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]))-0.0035f*(cons_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4])))*dxinv0);
      }
    }
  }
}

void hypterm_2 (double * __restrict flux_0,
                double * __restrict flux_1,
                double * __restrict flux_2,
                double * __restrict flux_3,
                double * __restrict flux_4,
                const double * __restrict cons_1,
                const double * __restrict cons_2,
                const double * __restrict cons_3,
                const double * __restrict cons_4, 
                const double * __restrict q_1,
                const double * __restrict q_2,
                const double * __restrict q_3,
                const double * __restrict q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int k = 4; k < N-4; k++) {
    for (int j = 4; j < N-4; j++) {
      for (int i = 4; i < N-4; i++) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i])-0.2f*(cons_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i])+0.038f*(cons_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_1[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_1[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_1[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_1[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_1[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_1[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_1[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]))-0.2f*(cons_2[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]))+0.038f*(cons_2[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]))-0.0035f*(cons_2[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i])))*dxinv1;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_3[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_3[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_3[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_3[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_3[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_3[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_3[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]))-0.2f*(cons_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]))+0.038f*(cons_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]))-0.0035f*(cons_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i])))*dxinv2;
      }
    }
  }
}

void hypterm_3 (double * __restrict flux_0,
                double * __restrict flux_1,
                double * __restrict flux_2,
                double * __restrict flux_3,
                double * __restrict flux_4,
                const double * __restrict cons_1,
                const double * __restrict cons_2,
                const double * __restrict cons_3,
                const double * __restrict cons_4, 
                const double * __restrict q_1,
                const double * __restrict q_2,
                const double * __restrict q_3,
                const double * __restrict q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int k = 4; k < N-4; k++) {
    for (int j = 4; j < N-4; j++) {
      for (int i = 4; i < N-4; i++) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i])-0.2f*(cons_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i])+0.038f*(cons_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_1[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_1[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_1[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_1[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_1[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_1[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_1[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_2[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_2[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_2[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_2[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_2[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_2[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_2[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]))-0.2f*(cons_3[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]))+0.038f*(cons_3[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]))-0.0035f*(cons_3[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i])))*dxinv2;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]))-0.2f*(cons_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]))+0.038f*(cons_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]))-0.0035f*(cons_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i])))*dxinv1;
      }
    }
  }
}

extern "C" void offload (double *h_flux_0, double *h_flux_1, double *h_flux_2, double *h_flux_3, double *h_flux_4, double *h_cons_1, double *h_cons_2, double *h_cons_3, double *h_cons_4, double *h_q_1, double *h_q_2, double *h_q_3, double *h_q_4, double dxinv0, double dxinv1, double dxinv2, int L, int M, int N, int repeat) {

  size_t vol = L*M*N;
  size_t vol_size = sizeof(double) * vol;

  double *flux_0 = (double*) malloc (vol_size);
  double *flux_1 = (double*) malloc (vol_size);
  double *flux_2 = (double*) malloc (vol_size);
  double *flux_3 = (double*) malloc (vol_size);
  double *flux_4 = (double*) malloc (vol_size);

  long t1 = 0, t2 = 0, t3 = 0;

  double *cons_1 = h_cons_1;
  double *cons_2 = h_cons_2;
  double *cons_3 = h_cons_3;
  double *cons_4 = h_cons_4;
  double *q_1 = h_q_1;
  double *q_2 = h_q_2;
  double *q_3 = h_q_3;
  double *q_4 = h_q_4;

  #pragma omp target data map(from:flux_0[0:vol], \
                                   flux_1[0:vol], \
                                   flux_2[0:vol], \
                                   flux_3[0:vol], \
                                   flux_4[0:vol]) \
                          map(to:cons_1[0:vol], \
                                 cons_2[0:vol], \
                                 cons_3[0:vol], \
                                 cons_4[0:vol], \
                                 q_1[0:vol],\
                                 q_2[0:vol],\
                                 q_3[0:vol],\
                                 q_4[0:vol])
  {
    for (int i = 0; i < repeat; i++) {
      memcpy (flux_0, h_flux_0, vol_size);
      memcpy (flux_1, h_flux_1, vol_size);
      memcpy (flux_2, h_flux_2, vol_size);
      memcpy (flux_3, h_flux_3, vol_size);
      memcpy (flux_4, h_flux_4, vol_size);
      #pragma omp target update to (flux_0[0:vol])
      #pragma omp target update to (flux_1[0:vol])
      #pragma omp target update to (flux_2[0:vol])
      #pragma omp target update to (flux_3[0:vol])

      auto start = std::chrono::steady_clock::now();
      hypterm_1(flux_0, flux_1, flux_2, flux_3, flux_4,
                cons_1, cons_2, cons_3, cons_4,
                q_1, q_2, q_3, q_4,
                dxinv0, dxinv1, dxinv2, L, M, N);
      auto end = std::chrono::steady_clock::now();
      t1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      start = std::chrono::steady_clock::now();
      hypterm_2(flux_0, flux_1, flux_2, flux_3, flux_4,
                cons_1, cons_2, cons_3, cons_4,
                q_1, q_2, q_3, q_4,
                dxinv0, dxinv1, dxinv2, L, M, N);
      end = std::chrono::steady_clock::now();
      t2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      start = std::chrono::steady_clock::now();
      hypterm_3(flux_0, flux_1, flux_2, flux_3, flux_4,
                cons_1, cons_2, cons_3, cons_4,
                q_1, q_2, q_3, q_4,
                dxinv0, dxinv1, dxinv2, L, M, N);
      end = std::chrono::steady_clock::now();
      t3 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    printf("Average kernel execution time (k1): %f (ms)\n", t1 * 1e-6 / repeat);
    printf("Average kernel execution time (k2): %f (ms)\n", t2 * 1e-6 / repeat);
    printf("Average kernel execution time (k3): %f (ms)\n", t3 * 1e-6 / repeat);
  }

  memcpy (h_flux_0, flux_0, vol_size);
  memcpy (h_flux_1, flux_1, vol_size);
  memcpy (h_flux_2, flux_2, vol_size);
  memcpy (h_flux_3, flux_3, vol_size);
  memcpy (h_flux_4, flux_4, vol_size);

  free(flux_0);
  free(flux_1);
  free(flux_2);
  free(flux_3);
  free(flux_4);
}

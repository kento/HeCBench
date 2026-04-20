/*
KMC kernel matrix computation for SVM.
OpenMP target offloading port.

Usage: ./main [ntv [len_tv [repeat]]]
*/

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[])
{
    int ntv    = 100;
    int len_tv = 50;
    int repeat = 10;

    if (argc > 1) ntv    = atoi(argv[1]);
    if (argc > 2) len_tv = atoi(argv[2]);
    if (argc > 3) repeat = atoi(argv[3]);

    const float gamma_val = 0.5f;

    // Training matrix: tva[i*len_tv + k] = feature k of vector i
    float*  d_tva   = (float*)malloc(ntv * len_tv * sizeof(float));
    float*  d_vtm   = (float*)malloc(len_tv * sizeof(float));
    float*  d_dp    = (float*)malloc(ntv * sizeof(float));
    double* d_tv_sq = (double*)malloc(ntv * sizeof(double));
    double* d_vfg   = (double*)malloc(ntv * sizeof(double));

    // Initialize random training data
    srand(42);
    for (int i = 0; i < ntv; i++) {
        double sq = 0.0;
        for (int k = 0; k < len_tv; k++) {
            float val = (float)rand() / (float)RAND_MAX;
            d_tva[i * len_tv + k] = val;
            sq += (double)val * (double)val;
        }
        d_tv_sq[i] = sq;
    }

    #pragma omp target enter data \
        map(to: d_tva[0:ntv*len_tv], d_tv_sq[0:ntv]) \
        map(alloc: d_vtm[0:len_tv], d_dp[0:ntv], d_vfg[0:ntv])

    double total_inner_ns = 0.0;
    double total_outer_ns = 0.0;

    const int ltv = len_tv;
    const int ntv_ = ntv;

    for (int rep = 0; rep < repeat; rep++) {
        auto t_offload_start = std::chrono::steady_clock::now();
        double rep_inner_ns = 0.0;

        for (int trvei = 0; trvei < ntv; trvei++) {
            const int tv_idx = trvei;

            // Copy current training vector into vtm
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int k = 0; k < ltv; k++) {
                d_vtm[k] = d_tva[tv_idx * ltv + k];
            }

            auto t1 = std::chrono::steady_clock::now();

            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int i = 0; i < ntv_; i++) {
                float s = 0.0f;
                for (int k = 0; k < ltv; k++)
                    s += d_tva[i * ltv + k] * d_vtm[k];
                d_dp[i] = s;
            }

            auto t2 = std::chrono::steady_clock::now();
            rep_inner_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

            const int trv = trvei;
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int ic = 0; ic < ntv_; ic++) {
                d_vfg[ic] = exp(-(double)gamma_val *
                             (d_tv_sq[trv] + d_tv_sq[ic] - 2.0 * (double)d_dp[ic]));
            }
        }

        auto t_offload_end = std::chrono::steady_clock::now();
        double offload_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                t_offload_end - t_offload_start).count();

        total_inner_ns += rep_inner_ns;
        total_outer_ns += offload_ns;
    }

    double avg_inner_ns = total_inner_ns / repeat;
    double avg_outer_ns = total_outer_ns / repeat;

    printf("Average kernel matrix offload time: %lf (us)\n",
           (avg_inner_ns * 1e-3) / ntv);
    printf("Total kernel matrix execution time: %lf (us)\n",
           avg_outer_ns * 1e-3);

    #pragma omp target exit data \
        map(delete: d_tva[0:ntv*len_tv], d_tv_sq[0:ntv], d_vtm[0:len_tv], d_dp[0:ntv], d_vfg[0:ntv])

    free(d_tva); free(d_vtm); free(d_dp); free(d_tv_sq); free(d_vfg);
    return 0;
}

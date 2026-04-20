#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

static float rand_float() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                                  const float* inp, const float* weight, const float* bias,
                                  int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* x = inp + i * C;
        float m = 0.0f;
        for (int j = 0; j < C; j++) m += x[j];
        m /= C;
        float v = 0.0f;
        for (int j = 0; j < C; j++) { float d = x[j] - m; v += d * d; }
        v /= C;
        float s = 1.0f / sqrtf(v + 1e-5f);
        mean[i] = m;
        rstd[i] = s;
        float* o = out + i * C;
        for (int j = 0; j < C; j++)
            o[j] = (s * (x[j] - m)) * weight[j] + bias[j];
    }
}

int main(int argc, char** argv) {
    int kernel_num = 1;
    if (argc > 1) kernel_num = atoi(argv[1]);
    printf("Using kernel 1 (OpenMP)\n");

    srand(0);
    const int B = 8, T = 1024, C = 768;
    const int N = B * T;

    std::vector<float> h_inp(N * C), h_weight(C), h_bias(C);
    std::vector<float> ref_out(N * C), ref_mean(N), ref_rstd(N);

    for (int i = 0; i < N * C; i++) h_inp[i]    = rand_float();
    for (int i = 0; i < C;     i++) h_weight[i] = rand_float();
    for (int i = 0; i < C;     i++) h_bias[i]   = rand_float();

    layernorm_forward_cpu(ref_out.data(), ref_mean.data(), ref_rstd.data(),
                          h_inp.data(), h_weight.data(), h_bias.data(), N, C);

    float* inp    = (float*)malloc(N * C * sizeof(float));
    float* weight = (float*)malloc(C * sizeof(float));
    float* bias   = (float*)malloc(C * sizeof(float));
    float* out    = (float*)malloc(N * C * sizeof(float));
    float* d_mean = (float*)malloc(N * sizeof(float));
    float* d_rstd = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N * C; i++) inp[i]    = h_inp[i];
    for (int i = 0; i < C;     i++) weight[i] = h_weight[i];
    for (int i = 0; i < C;     i++) bias[i]   = h_bias[i];

    #pragma omp target enter data \
        map(to: inp[0:N*C], weight[0:C], bias[0:C]) \
        map(alloc: out[0:N*C], d_mean[0:N], d_rstd[0:N])

    const int NC = N * C;
    const int Cv  = C;
    const int Nv  = N;

    auto run_kernel = [&]() {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < Nv; i++) {
            float m = 0.0f;
            for (int j = 0; j < Cv; j++) m += inp[i * Cv + j];
            m /= Cv;
            float v = 0.0f;
            for (int j = 0; j < Cv; j++) {
                float d = inp[i * Cv + j] - m;
                v += d * d;
            }
            v /= Cv;
            float s = 1.0f / sqrtf(v + 1e-5f);
            d_mean[i] = m;
            d_rstd[i] = s;
            for (int j = 0; j < Cv; j++)
                out[i * Cv + j] = (s * (inp[i * Cv + j] - m)) * weight[j] + bias[j];
        }
    };

    // Warmup
    run_kernel();

    // Correctness check
    printf("Checking correctness...\n");
    #pragma omp target update from(out[0:N*C], d_mean[0:N], d_rstd[0:N])

    int nfaults = 0;
    for (int i = 0; i < N * C && nfaults < 10; i++) {
        float tol = 1e-4f + fabsf(ref_out[i]) * 1.19e-7f;
        if (fabsf(out[i] - ref_out[i]) > tol) {
            printf("Mismatch out[%d]: got %f expected %f\n", i, out[i], ref_out[i]);
            nfaults++;
        }
    }
    for (int i = 0; i < N && nfaults < 10; i++) {
        if (fabsf(d_mean[i] - ref_mean[i]) > 1e-4f) {
            printf("Mismatch mean[%d]: got %f expected %f\n", i, d_mean[i], ref_mean[i]);
            nfaults++;
        }
        if (fabsf(d_rstd[i] - ref_rstd[i]) > 1e-4f) {
            printf("Mismatch rstd[%d]: got %f expected %f\n", i, d_rstd[i], ref_rstd[i]);
            nfaults++;
        }
    }
    if (nfaults > 0) {
        #pragma omp target exit data map(delete: inp[0:N*C], weight[0:C], bias[0:C], \
            out[0:N*C], d_mean[0:N], d_rstd[0:N])
        free(inp); free(weight); free(bias); free(out); free(d_mean); free(d_rstd);
        return 1;
    }

    printf("All results match. Starting benchmarks.\n");

    const int repeat = 2000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < repeat; n++) run_kernel();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;

    long bytes = (long)(2 * N * C + 2 * C + 2 * N) * sizeof(float);
    double gb_s = (bytes * 1e-9) / (ms * 1e-3);

    printf("Average kernel time: %.4f ms\n", ms);
    printf("Memory bandwidth: %.2f GB/s\n", gb_s);

    #pragma omp target exit data map(delete: inp[0:N*C], weight[0:C], bias[0:C], \
        out[0:N*C], d_mean[0:N], d_rstd[0:N])
    free(inp); free(weight); free(bias); free(out); free(d_mean); free(d_rstd);
    return 0;
}

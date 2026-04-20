// mrg32k3a - OpenMP target offloading port
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr double MRG_M1   = 4294967087.0;
static constexpr double MRG_M2   = 4294944443.0;
static constexpr double MRG_A12  = 1403580.0;
static constexpr double MRG_A13N = 810728.0;
static constexpr double MRG_A21  = 527612.0;
static constexpr double MRG_A23N = 1370589.0;

// Host-only sequential reference generator
static void run_on_host(int n, unsigned long long seed, std::vector<float>& out) {
    double s10 = 1.0 + (double)(seed % (unsigned long long)(MRG_M1 - 2));
    double s11 = 12345.0;
    double s12 = 67890.0;
    double s20 = 1.0 + (double)((seed * 6364136223846793005ULL + 1442695040888963407ULL)
                                % (unsigned long long)(MRG_M2 - 2));
    double s21 = 24680.0;
    double s22 = 11111.0;

    for (int i = 0; i < n; ++i) {
        double p1 = MRG_A12 * s11 - MRG_A13N * s10;
        p1 = std::fmod(p1, MRG_M1);
        if (p1 < 0.0) p1 += MRG_M1;

        double p2 = MRG_A21 * s22 - MRG_A23N * s20;
        p2 = std::fmod(p2, MRG_M2);
        if (p2 < 0.0) p2 += MRG_M2;

        s10 = s11; s11 = s12; s12 = p1;
        s20 = s21; s21 = s22; s22 = p2;

        double u = (p1 - p2) / MRG_M1;
        if (p1 <= p2) u += 1.0;
        out[i] = (float)u;
    }
}

// Device functions: hash and per-element generator
#pragma omp declare target

static unsigned long long mrg_hash(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static float mrg32k3a_single(int idx, unsigned long long seed) {
    unsigned long long h0 = mrg_hash(seed ^ (unsigned long long)(idx) * 6364136223846793005ULL);
    unsigned long long h1 = mrg_hash(h0 + 1);
    unsigned long long h2 = mrg_hash(h1 + 2);
    unsigned long long h3 = mrg_hash(h2 + 3);
    unsigned long long h4 = mrg_hash(h3 + 4);
    unsigned long long h5 = mrg_hash(h4 + 5);

    // Clamp into valid MRG state ranges (inlined from original lambdas)
    double s10 = 1.0 + (double)(h0 % (unsigned long long)(MRG_M1 - 2));
    double s11 = 1.0 + (double)(h1 % (unsigned long long)(MRG_M1 - 2));
    double s12 = 1.0 + (double)(h2 % (unsigned long long)(MRG_M1 - 2));
    double s20 = 1.0 + (double)(h3 % (unsigned long long)(MRG_M2 - 2));
    double s21 = 1.0 + (double)(h4 % (unsigned long long)(MRG_M2 - 2));
    double s22 = 1.0 + (double)(h5 % (unsigned long long)(MRG_M2 - 2));

    double p1 = MRG_A12 * s11 - MRG_A13N * s10;
    p1 = fmod(p1, MRG_M1);
    if (p1 < 0.0) p1 += MRG_M1;

    double p2 = MRG_A21 * s22 - MRG_A23N * s20;
    p2 = fmod(p2, MRG_M2);
    if (p2 < 0.0) p2 += MRG_M2;

    double u = (p1 - p2) / MRG_M1;
    if (p1 <= p2) u += 1.0;
    return (float)u;
}

#pragma omp end declare target

static void run_on_device(int n, unsigned long long seed, float* d_out) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++)
        d_out[i] = mrg32k3a_single(i, seed);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <number of pseudorandom numbers> <repeat>\n", argv[0]);
        return 1;
    }
    const int  n      = atoi(argv[1]);
    const int  repeat = atoi(argv[2]);
    const unsigned long long seed = 1234ULL;

    std::vector<float> h_data(n, 0.f);
    float* d_data = (float*)malloc(n * sizeof(float));
    #pragma omp target enter data map(alloc: d_data[0:n])

    // Warmup / initial run for both host and device
    run_on_host(n, seed, h_data);
    run_on_device(n, seed, d_data);

    // Time host
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_on_host(n, seed, h_data);
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time on host: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);

    // Time device
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_on_device(n, seed, d_data);
    t1 = std::chrono::steady_clock::now();
    printf("Average execution time on device: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);

    // Bring device results to host for validation
    #pragma omp target update from(d_data[0:n])

    bool ok_host = true, ok_dev = true;
    for (int i = 0; i < n; ++i) {
        if (h_data[i] < 0.f || h_data[i] >= 1.f) { ok_host = false; break; }
    }
    for (int i = 0; i < n; ++i) {
        if (d_data[i] < 0.f || d_data[i] >= 1.f) { ok_dev = false; break; }
    }
    printf("%s\n", (ok_host && ok_dev) ? "PASS" : "FAIL");

    #pragma omp target exit data map(delete: d_data[0:n])
    free(d_data);
    return 0;
}

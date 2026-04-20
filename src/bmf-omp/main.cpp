// Binary Matrix Factorization (BMF) benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>

static constexpr int   M          = 1000;
static constexpr int   N          = 1000;
static constexpr int   D          = 20;
static constexpr int   MAX_ITER   = 100;
static constexpr float TEMP_START = 1.0f;
static constexpr float TEMP_END   = 0.01f;
static constexpr float TEMP_DECAY = 0.9f;

static constexpr int ROW_GROUPS = (M + 31) / 32;
static constexpr int C_FLAT     = ROW_GROUPS * N;

#pragma omp declare target
uint32_t xorshift32(uint32_t s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

int get_C(const uint32_t* C_d, int i, int j)
{
    return (int)((C_d[(i / 32) * N + j] >> (i % 32)) & 1u);
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
    uint32_t* A_d = (uint32_t*)malloc(M * sizeof(uint32_t));
    uint32_t* B_d = (uint32_t*)malloc(N * sizeof(uint32_t));
    uint32_t* C_d = (uint32_t*)malloc(C_FLAT * sizeof(uint32_t));

    const uint32_t dmask = (D < 32) ? ((1u << D) - 1u) : ~0u;

    {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<uint32_t> udist(0, ~0u);
        uint32_t* Ah = (uint32_t*)malloc(M * sizeof(uint32_t));
        uint32_t* Bh = (uint32_t*)malloc(N * sizeof(uint32_t));

        for (int i = 0; i < M; ++i) Ah[i] = udist(rng) & dmask;
        for (int j = 0; j < N; ++j) Bh[j] = udist(rng) & dmask;

        memset(C_d, 0, C_FLAT * sizeof(uint32_t));
        for (int i = 0; i < M; ++i) {
            uint32_t Ai = Ah[i];
            for (int j = 0; j < N; ++j)
                if (Ai & Bh[j])
                    C_d[(i / 32) * N + j] |= (1u << (i % 32));
        }
        for (int i = 0; i < M; ++i) Ah[i] = udist(rng) & dmask;
        for (int j = 0; j < N; ++j) Bh[j] = udist(rng) & dmask;
        memcpy(A_d, Ah, M * sizeof(uint32_t));
        memcpy(B_d, Bh, N * sizeof(uint32_t));
        free(Ah); free(Bh);
    }

    #pragma omp target enter data map(alloc: A_d[0:M], B_d[0:N], C_d[0:C_FLAT])
    #pragma omp target update to(A_d[0:M], B_d[0:N], C_d[0:C_FLAT])

    // Compute initial error
    auto compute_error = [&]() -> int {
        int err = 0;
        #pragma omp target teams distribute parallel for reduction(+:err) thread_limit(256)
        for (int i = 0; i < M; i++) {
            const uint32_t Ai = A_d[i];
            int local = 0;
            for (int j = 0; j < N; ++j) {
                const int prod = (Ai & B_d[j]) ? 1 : 0;
                const int C_ij = get_C(C_d, i, j);
                local += (prod != C_ij) ? 1 : 0;
            }
            err += local;
        }
        return err;
    };

    const int init_error = compute_error();
    printf("Initial error: %d / %d  (%.2f%%)\n",
           init_error, M * N, 100.0 * init_error / (M * N));

    auto t0 = std::chrono::high_resolution_clock::now();

    float temp = TEMP_START;
    for (int iter = 0; iter < MAX_ITER && temp > TEMP_END; ++iter) {
        const uint32_t seed_A = xorshift32((uint32_t)(iter * 6364136223846793005u + 1));
        const uint32_t seed_B = xorshift32(seed_A + 0xDEADBEEFu);

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < M; i++) {
            uint32_t rng = xorshift32(seed_A ^ (uint32_t)(i + 1));
            rng = xorshift32(rng);
            const int bit = (int)(rng % (uint32_t)D);
            const uint32_t Ai     = A_d[i];
            const uint32_t Ai_new = Ai ^ (1u << bit);
            int delta = 0;
            for (int j = 0; j < N; ++j) {
                const int old_prod = (Ai     & B_d[j]) ? 1 : 0;
                const int new_prod = (Ai_new & B_d[j]) ? 1 : 0;
                if (old_prod == new_prod) continue;
                const int C_ij = get_C(C_d, i, j);
                delta += (new_prod != C_ij) - (old_prod != C_ij);
            }
            bool accept = (delta <= 0);
            if (!accept) {
                rng = xorshift32(rng ^ (uint32_t)delta);
                const float p = expf(-(float)delta / temp);
                const float r = (float)(rng >> 8) / (float)(1u << 24);
                accept = (r < p);
            }
            if (accept) A_d[i] = Ai_new;
        }

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int j = 0; j < N; j++) {
            uint32_t rng = xorshift32(seed_B ^ (uint32_t)(j + 1));
            rng = xorshift32(rng);
            const int bit = (int)(rng % (uint32_t)D);
            const uint32_t Bj     = B_d[j];
            const uint32_t Bj_new = Bj ^ (1u << bit);
            int delta = 0;
            for (int i = 0; i < M; ++i) {
                const int old_prod = (A_d[i] & Bj    ) ? 1 : 0;
                const int new_prod = (A_d[i] & Bj_new) ? 1 : 0;
                if (old_prod == new_prod) continue;
                const int C_ij = get_C(C_d, i, j);
                delta += (new_prod != C_ij) - (old_prod != C_ij);
            }
            bool accept = (delta <= 0);
            if (!accept) {
                rng = xorshift32(rng ^ (uint32_t)delta);
                const float p = expf(-(float)delta / temp);
                const float r = (float)(rng >> 8) / (float)(1u << 24);
                accept = (r < p);
            }
            if (accept) B_d[j] = Bj_new;
        }

        temp *= TEMP_DECAY;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    const int final_error = compute_error();
    printf("BMF OpenMP benchmark\n");
    printf("Matrix %dx%d, rank %d, SA iterations %d\n", M, N, D, MAX_ITER);
    printf("Final error  : %d / %d  (%.2f%%)\n",
           final_error, M * N, 100.0 * final_error / (M * N));
    printf("Total time   : %.3f s\n", elapsed);
    printf("Avg per iter : %.3f ms\n", elapsed * 1e3 / MAX_ITER);

    #pragma omp target exit data map(delete: A_d[0:M], B_d[0:N], C_d[0:C_FLAT])
    free(A_d); free(B_d); free(C_d);
    return 0;
}

// Matrix Factorization SGD - OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[]) {
    int num_users   = (argc > 1) ? atoi(argv[1]) : 1000;
    int num_items   = (argc > 2) ? atoi(argv[2]) : 1000;
    int num_ratings = (argc > 3) ? atoi(argv[3]) : 50000;
    int k           = (argc > 4) ? atoi(argv[4]) : 128;
    int num_iters   = (argc > 5) ? atoi(argv[5]) : 5;
    int repeat      = (argc > 6) ? atoi(argv[6]) : 1;

    const float lrate  = 0.001f;
    const float lambda = 0.001f;

    printf("MF-SGD: users=%d items=%d ratings=%d k=%d iters=%d repeat=%d\n",
           num_users, num_items, num_ratings, k, num_iters, repeat);

    // Allocate and initialise rating data on host, then map to device
    int*   d_users   = (int*)malloc(num_ratings * sizeof(int));
    int*   d_items   = (int*)malloc(num_ratings * sizeof(int));
    float* d_ratings = (float*)malloc(num_ratings * sizeof(float));

    srand(42);
    for (int i = 0; i < num_ratings; i++) {
        d_users[i]   = rand() % num_users;
        d_items[i]   = rand() % num_items;
        d_ratings[i] = 1.0f + (float)(rand() % 5);
    }
    #pragma omp target enter data map(alloc: d_users[0:num_ratings], \
                                             d_items[0:num_ratings], \
                                             d_ratings[0:num_ratings])
    #pragma omp target update to(d_users[0:num_ratings], \
                                 d_items[0:num_ratings], \
                                 d_ratings[0:num_ratings])

    // Latent factor matrices
    int pk_size = num_users * k;
    int qk_size = num_items * k;

    float* d_P  = (float*)malloc(pk_size * sizeof(float));
    float* d_Q  = (float*)malloc(qk_size * sizeof(float));
    float* d_P0 = (float*)malloc(pk_size * sizeof(float));
    float* d_Q0 = (float*)malloc(qk_size * sizeof(float));

    srand(123);
    for (int i = 0; i < pk_size; i++)
        d_P0[i] = 0.1f * ((float)rand() / RAND_MAX);
    for (int i = 0; i < qk_size; i++)
        d_Q0[i] = 0.1f * ((float)rand() / RAND_MAX);

    #pragma omp target enter data map(alloc: d_P[0:pk_size], d_Q[0:qk_size], \
                                             d_P0[0:pk_size], d_Q0[0:qk_size])
    #pragma omp target update to(d_P0[0:pk_size], d_Q0[0:qk_size])

    double total_time = 0.0;

    for (int r = 0; r < repeat; r++) {
        // Device-to-device copy: reset P and Q from their initial values
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < pk_size; i++) d_P[i] = d_P0[i];

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < qk_size; i++) d_Q[i] = d_Q0[i];

        double t_start = omp_get_wtime();

        for (int iter = 0; iter < num_iters; iter++) {
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int idx = 0; idx < num_ratings; idx++) {
                const int   u    = d_users[idx];
                const int   v    = d_items[idx];
                const float r_uv = d_ratings[idx];

                float predict = 0.0f;
                for (int f = 0; f < k; f++)
                    predict += d_P[u * k + f] * d_Q[v * k + f];

                const float err = r_uv - predict;

                for (int f = 0; f < k; f++) {
                    const float pu = d_P[u * k + f];
                    const float qv = d_Q[v * k + f];
                    const float dp = lrate * (err * qv - lambda * pu);
                    const float dq = lrate * (err * pu - lambda * qv);
                    #pragma omp atomic update
                    d_P[u * k + f] += dp;
                    #pragma omp atomic update
                    d_Q[v * k + f] += dq;
                }
            }
        }

        total_time += omp_get_wtime() - t_start;
    }

    const double avg_ms = total_time * 1000.0 / repeat;
    printf("Average SGD time (%d iters): %.3f ms\n", num_iters, avg_ms);

    // Compute RMSE on device
    float rmse = 0.0f;
    #pragma omp target teams distribute parallel for reduction(+:rmse) thread_limit(256)
    for (int idx = 0; idx < num_ratings; idx++) {
        const int   u    = d_users[idx];
        const int   v    = d_items[idx];
        const float r_uv = d_ratings[idx];
        float pred = 0.0f;
        for (int f = 0; f < k; f++)
            pred += d_P[u * k + f] * d_Q[v * k + f];
        const float e = r_uv - pred;
        rmse += e * e;
    }
    rmse = sqrtf(rmse / num_ratings);
    printf("Final RMSE: %.6f\n", rmse);

    #pragma omp target exit data map(delete: d_users[0:num_ratings], \
                                             d_items[0:num_ratings], \
                                             d_ratings[0:num_ratings])
    #pragma omp target exit data map(delete: d_P[0:pk_size], d_Q[0:qk_size], \
                                             d_P0[0:pk_size], d_Q0[0:qk_size])

    free(d_users); free(d_items); free(d_ratings);
    free(d_P); free(d_Q); free(d_P0); free(d_Q0);
    return 0;
}

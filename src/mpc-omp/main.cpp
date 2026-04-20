// MPC Delta Compression - OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

int main(int argc, char* argv[]) {
    const int repeat = (argc > 1) ? atoi(argv[1]) : 1;
    const int n      = (argc > 2) ? atoi(argv[2]) : 1000000;

    printf("MPC compression (simplified delta-XOR): n=%d doubles, repeat=%d\n", n, repeat);

    // Initialise input on host
    uint64_t* d_input = (uint64_t*)malloc(n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) {
        double val = sin(i * 0.001) * 1e6 + cos(i * 0.0013) * 1e4;
        memcpy(&d_input[i], &val, sizeof(double));
    }
    #pragma omp target enter data map(alloc: d_input[0:n])
    #pragma omp target update to(d_input[0:n])

    uint64_t* d_delta  = (uint64_t*)malloc(n * sizeof(uint64_t));
    int*      d_nbytes = (int*)malloc(n * sizeof(int));
    int*      d_offset = (int*)malloc(n * sizeof(int));
    int       packed_alloc = n * 8;
    uint8_t*  d_packed = (uint8_t*)malloc(packed_alloc * sizeof(uint8_t));

    #pragma omp target enter data map(alloc: d_delta[0:n], d_nbytes[0:n], \
                                             d_offset[0:n], d_packed[0:packed_alloc])

    double  total_time  = 0.0;
    int64_t packed_size = 0;

    for (int rep = 0; rep < repeat; rep++) {
        double t_start = omp_get_wtime();

        // Step 1: compute XOR delta and per-element byte count on device
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < n; i++) {
            const uint64_t cur   = d_input[i];
            const uint64_t prev  = (i > 0) ? d_input[i - 1] : 0ULL;
            const uint64_t delta = cur ^ prev;
            d_delta[i] = delta;
            int cnt = 0;
            for (int b = 0; b < 8; b++)
                if ((delta >> (8 * b)) & 0xFF) cnt++;
            d_nbytes[i] = cnt;
        }

        // Step 2: bring byte counts to host and compute prefix-sum there
        #pragma omp target update from(d_nbytes[0:n])
        {
            int acc = 0;
            for (int i = 0; i < n; i++) {
                d_offset[i] = acc;
                acc += d_nbytes[i];
            }
            packed_size = acc;
        }
        #pragma omp target update to(d_offset[0:n])

        // Step 3: pack non-zero bytes into the output buffer on device
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < n; i++) {
            const uint64_t delta = d_delta[i];
            int out = d_offset[i];
            for (int b = 0; b < 8; b++) {
                const uint8_t byte = (uint8_t)((delta >> (8 * b)) & 0xFF);
                if (byte) d_packed[out++] = byte;
            }
        }

        total_time += omp_get_wtime() - t_start;
    }

    const double avg_ms    = total_time * 1000.0 / repeat;
    const double ratio     = (double)(n * 8) / packed_size;
    const double throughput = (double)n * 8 / (total_time / repeat) / 1e9;

    printf("Input size      : %d doubles = %d bytes\n", n, n * 8);
    printf("Packed size     : %ld bytes\n", (long)packed_size);
    printf("Compression ratio: %.3f:1\n", ratio);
    printf("Average time    : %.3f ms\n", avg_ms);
    printf("Throughput      : %.3f GB/s\n", throughput);

    #pragma omp target exit data map(delete: d_input[0:n], d_delta[0:n], \
                                             d_nbytes[0:n], d_offset[0:n], \
                                             d_packed[0:packed_alloc])
    free(d_input); free(d_delta); free(d_nbytes); free(d_offset); free(d_packed);
    return 0;
}

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define P1 55
#define P2 119
#define P3 179
#define P4 256

typedef unsigned int uint32_t_l;

static void LFIB4(uint32_t_l n, uint32_t_l *x) {
    for (uint32_t_l k = P4; k < n; k++) {
        x[k] = x[k-P1] + x[k-P2] + x[k-P3] + x[k-P4];
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ./main <n>\n");
        return 1;
    }

    uint32_t_l n = (uint32_t_l)atoi(argv[1]);

    srand(1234);

    uint32_t_l *x      = (uint32_t_l*)malloc(n * sizeof(uint32_t_l));
    uint32_t_l *d_x    = (uint32_t_l*)malloc(n * sizeof(uint32_t_l));

    #pragma omp target enter data map(alloc: d_x[0:n])

    for (uint32_t_l r = 16; r <= 4096; r = r * 2) {
        uint32_t_l s = n / r;
        s -= (s % P4 == 0 ? 0 : s % P4);
        if (s == 0) break;
        while (s * r < n) r++;

        printf("n=%u r=%u s=%u\n", n, r, s);

        uint32_t_l *seed = (uint32_t_l*)malloc(P4 * sizeof(uint32_t_l));
        for (uint32_t_l k = 0; k < P4; k++) x[k] = seed[k] = (uint32_t_l)rand();

        // Host computation
        auto h_start = std::chrono::steady_clock::now();
        LFIB4(n, x);
        auto h_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> host_time = h_end - h_start;

        // Device computation
        for (uint32_t_l k = 0; k < P4; k++) d_x[k] = seed[k];
        #pragma omp target update to(d_x[0:n])

        auto d_start = std::chrono::steady_clock::now();
        // LFIB4 has strict sequential dependency, run as single work-item
        #pragma omp target teams num_teams(1) thread_limit(1)
        {
            for (uint32_t_l k = P4; k < n; k++) {
                d_x[k] = d_x[k-P1] + d_x[k-P2] + d_x[k-P3] + d_x[k-P4];
            }
        }
        auto d_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> device_time = d_end - d_start;

        // Verify
        #pragma omp target update from(d_x[0:n])
        bool ok = true;
        for (uint32_t_l i = 0; i < n; i++) {
            if (x[i] != d_x[i]) { ok = false; break; }
        }

        double speedup = (device_time.count() > 0.0)
                         ? host_time.count() / device_time.count()
                         : 1.0;

        printf("r = %u | host time = %lf | device time = %lf | speedup = %.1f "
               "check = %s\n",
               r, host_time.count(), device_time.count(), speedup,
               ok ? "PASS" : "FAIL");

        free(seed);
    }

    #pragma omp target exit data map(delete: d_x[0:n])
    free(x); free(d_x);
    return 0;
}

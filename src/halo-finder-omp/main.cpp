/*
Friends-of-Friends (FoF) Halo Finder – simplified cosmology benchmark.
OpenMP target offloading port.

Algorithm:
  1. Generate nParticles random 3D positions in [0,1]^3
  2. Link particles within linking length b into groups (halos)
  3. Use union-find for group assignment
  4. Count halos and their sizes

Usage: ./main [nParticles [repeat]]
*/

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <climits>

#pragma omp declare target
static float lcg_rand(unsigned long long& state)
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
    int nPart  = 10000;
    int repeat = 1;
    if (argc > 1) nPart  = atoi(argv[1]);
    if (argc > 2) repeat = atoi(argv[2]);

    printf("nParticles=%d\n", nPart);

    const float mean_spacing = (float)cbrt(1.0 / nPart);
    const float b            = 0.2f * mean_spacing;
    const float b2           = b * b;
    printf("Linking length: %f\n", (double)b);

    float* d_x      = (float*)malloc(nPart * sizeof(float));
    float* d_y      = (float*)malloc(nPart * sizeof(float));
    float* d_z      = (float*)malloc(nPart * sizeof(float));
    int*   d_label  = (int*)malloc(nPart * sizeof(int));
    int*   d_haloSize = (int*)malloc(nPart * sizeof(int));

    #pragma omp target enter data map(alloc: d_x[0:nPart], d_y[0:nPart], d_z[0:nPart], \
                                             d_label[0:nPart], d_haloSize[0:nPart])

    // Generate random positions on device
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < nPart; i++) {
        unsigned long long state = (unsigned long long)i * 6364136223846793005ULL + 1;
        d_x[i] = lcg_rand(state);
        d_y[i] = lcg_rand(state);
        d_z[i] = lcg_rand(state);
    }

    double total_ms = 0.0;
    int    last_nHalos  = 0;
    int    last_largest = 0;

    for (int rep = 0; rep < repeat; rep++) {
        // Initialize union-find
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < nPart; i++) d_label[i] = i;

        auto t1 = std::chrono::steady_clock::now();

        // Pair-wise FoF linking
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < nPart; i++) {
            float xi = d_x[i], yi = d_y[i], zi = d_z[i];
            for (int j = i + 1; j < nPart; j++) {
                float dx = xi - d_x[j];
                float dy = yi - d_y[j];
                float dz = zi - d_z[j];
                if (dx*dx + dy*dy + dz*dz < b2) {
                    int ri = i, rj = j;
                    while (d_label[ri] != ri) ri = d_label[ri];
                    while (d_label[rj] != rj) rj = d_label[rj];
                    while (ri != rj) {
                        if (ri < rj) {
                            #pragma omp atomic capture
                            { int old = d_label[rj]; if (old == rj) { d_label[rj] = ri; } rj = old; }
                            while (d_label[rj] != rj) rj = d_label[rj];
                        } else {
                            #pragma omp atomic capture
                            { int old = d_label[ri]; if (old == ri) { d_label[ri] = rj; } ri = old; }
                            while (d_label[ri] != ri) ri = d_label[ri];
                        }
                    }
                }
            }
        }

        // Path compression
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < nPart; i++) {
            int root = d_label[i];
            while (d_label[root] != root) root = d_label[root];
            d_label[i] = root;
        }

        auto t2 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Count halo sizes
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < nPart; i++) d_haloSize[i] = 0;

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < nPart; i++) {
            #pragma omp atomic update
            d_haloSize[d_label[i]]++;
        }

        int nHalos = 0;
        #pragma omp target teams distribute parallel for reduction(+:nHalos) thread_limit(256)
        for (int i = 0; i < nPart; i++) {
            if (d_label[i] == i && d_haloSize[i] > 0) nHalos++;
        }

        int largest = 0;
        #pragma omp target teams distribute parallel for reduction(max:largest) thread_limit(256)
        for (int i = 0; i < nPart; i++) {
            if (d_haloSize[i] > largest) largest = d_haloSize[i];
        }

        last_nHalos  = nHalos;
        last_largest = largest;
    }

    printf("Found %d halos\n",        last_nHalos);
    printf("Largest halo: %d particles\n", last_largest);
    printf("Kernel time: %.3f (ms)\n", total_ms / repeat);

    #pragma omp target exit data map(delete: d_x[0:nPart], d_y[0:nPart], d_z[0:nPart], \
                                             d_label[0:nPart], d_haloSize[0:nPart])
    free(d_x); free(d_y); free(d_z); free(d_label); free(d_haloSize);
    return 0;
}

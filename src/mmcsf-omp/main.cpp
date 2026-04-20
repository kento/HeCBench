#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

struct COOTensor {
    int ndims;
    std::vector<int> dims;
    std::vector<int> inds0, inds1, inds2;
    std::vector<float> vals;
    int nnz() const { return (int)vals.size(); }
};

static bool load_tns(const std::string& fname, COOTensor& T) {
    std::ifstream f(fname);
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname.c_str()); return false; }
    std::string line;
    if (!std::getline(f, line)) return false;
    T.ndims = std::stoi(line);
    if (!std::getline(f, line)) return false;
    { std::istringstream ss(line); int d; while (ss >> d) T.dims.push_back(d); }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        int i0, i1, i2; float val;
        if (!(ss >> i0 >> i1 >> i2 >> val)) continue;
        T.inds0.push_back(i0 - 1); T.inds1.push_back(i1 - 1);
        T.inds2.push_back(i2 - 1); T.vals.push_back(val);
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 2) { printf("Usage: %s <input.tns> [R] [repeat]\n", argv[0]); return 1; }
    const char *tnsFile = argv[1];
    int R      = (argc > 2) ? atoi(argv[2]) : 32;
    int repeat = (argc > 3) ? atoi(argv[3]) : 1;

    COOTensor X;
    if (!load_tns(tnsFile, X)) return 1;
    const int nnz = X.nnz();
    if (X.ndims != 3) { fprintf(stderr, "Only 3-D tensors supported\n"); return 1; }
    int dim0 = X.dims[0], dim1 = X.dims[1], dim2 = X.dims[2];
    printf("Tensor: %d x %d x %d  nnz=%d  R=%d  repeat=%d\n", dim0, dim1, dim2, nnz, R, repeat);

    int*   d_i0 = (int*)  malloc(nnz * sizeof(int));
    int*   d_i1 = (int*)  malloc(nnz * sizeof(int));
    int*   d_i2 = (int*)  malloc(nnz * sizeof(int));
    float* d_v  = (float*)malloc(nnz * sizeof(float));
    for (int n = 0; n < nnz; n++) {
        d_i0[n] = X.inds0[n]; d_i1[n] = X.inds1[n];
        d_i2[n] = X.inds2[n]; d_v[n]  = X.vals[n];
    }
#pragma omp target enter data map(to: d_i0[0:nnz], d_i1[0:nnz], d_i2[0:nnz], d_v[0:nnz])

    float* U0 = (float*)malloc(dim0 * R * sizeof(float));
    float* U1 = (float*)malloc(dim1 * R * sizeof(float));
    float* U2 = (float*)malloc(dim2 * R * sizeof(float));
    srand(42);
    for (int i = 0; i < dim1 * R; i++) U1[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < dim2 * R; i++) U2[i] = (float)rand() / RAND_MAX;
#pragma omp target enter data map(alloc: U0[0:dim0*R]) \
    map(to: U1[0:dim1*R], U2[0:dim2*R])

    double totalTime = 0.0;
    for (int rep = 0; rep < repeat; rep++) {
        // Zero U0 on device
#pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < dim0 * R; i++) U0[i] = 0.0f;

        auto t0 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for thread_limit(256)
        for (int n = 0; n < nnz; n++) {
            int i = d_i0[n], j = d_i1[n], k = d_i2[n];
            float val = d_v[n];
            for (int r = 0; r < R; r++) {
                float upd = val * U1[j * R + r] * U2[k * R + r];
#pragma omp atomic update
                U0[i * R + r] += upd;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        totalTime += std::chrono::duration<double>(t1 - t0).count();
    }
    printf("Average MTTKRP time: %.6f s\n", totalTime / repeat);
    printf("Throughput: %.3f GFLOP/s\n", (2.0 * nnz * R) / (totalTime / repeat) * 1e-9);

#pragma omp target update from(U0[0:dim0*R])
    printf("U0[0..%d]:", (int)std::min(R, 4) - 1);
    for (int r = 0; r < std::min(R, 4); r++) printf(" %.4f", U0[r]);
    printf("\n");

#pragma omp target exit data map(delete: d_i0[0:nnz], d_i1[0:nnz], d_i2[0:nnz], d_v[0:nnz])
#pragma omp target exit data map(delete: U0[0:dim0*R], U1[0:dim1*R], U2[0:dim2*R])
    free(d_i0); free(d_i1); free(d_i2); free(d_v);
    free(U0); free(U1); free(U2);
    return 0;
}

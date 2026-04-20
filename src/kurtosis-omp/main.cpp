#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

// ── Data types ───────────────────────────────────────────────────────────────

struct storeElement {
    int           tag;
    int           metric;
    unsigned long long time;
    float         value;
};

struct kurtosisResult {
    int   count;
    float mean, m2, m3, m4;
};

// Parallel-safe binary combine
#pragma omp declare target
static kurtosisResult kurtosis_combine(const kurtosisResult& x, const kurtosisResult& y) {
    float xc = (float)x.count;
    float yc = (float)y.count;
    float countf  = xc + yc;
    if (countf == 0.0f) {
        kurtosisResult r; r.count = 0; r.mean = 0; r.m2 = 0; r.m3 = 0; r.m4 = 0;
        return r;
    }
    float count2  = countf * countf;
    float count3  = count2 * countf;

    float delta   = y.mean - x.mean;
    float delta2  = delta  * delta;
    float delta3  = delta2 * delta;
    float delta4  = delta3 * delta;

    kurtosisResult r;
    r.count = x.count + y.count;
    r.mean  = x.mean + delta * yc / countf;

    r.m2 = x.m2 + y.m2
         + delta2 * xc * yc / countf;

    r.m3 = x.m3 + y.m3
         + delta3 * xc * yc * (xc - yc) / count2
         + 3.0f  * delta * (xc * y.m2 - yc * x.m2) / countf;

    r.m4 = x.m4 + y.m4
         + delta4 * xc * yc * (xc * xc - xc * yc + yc * yc) / count3
         + 6.0f  * delta2 * (xc * xc * y.m2 + yc * yc * x.m2) / count2
         + 4.0f  * delta  * (xc * y.m3 - yc * x.m3) / countf;
    return r;
}
#pragma omp end declare target

// OpenMP reduction for kurtosisResult: use a simple parallel loop with
// manual tree reduction via shared memory approach. Since OpenMP doesn't
// support custom user-defined reductions for struct types portably on GPU,
// we use a two-step approach: parallel partial results + host combine.
// For simplicity, we compute element-wise and combine in a single target region.

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <elemCount> <repeat>\n";
        return 1;
    }

    const int elemCount = atoi(argv[1]);
    const int repeat    = atoi(argv[2]);

    storeElement* h_elem = new storeElement[elemCount];
    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dis(1.f, 2.f);
    for (int i = 0; i < elemCount; i++) {
        h_elem[i].tag    = i;
        h_elem[i].metric = 0;
        h_elem[i].time   = (unsigned long long)i;
        h_elem[i].value  = dis(gen);
    }

    float*  d_values = (float*)malloc(elemCount * sizeof(float));
    for (int i = 0; i < elemCount; i++) d_values[i] = h_elem[i].value;

    // We store partial results per thread group on host
    // Use a flat array of partial kurtosis results
    const int nGroups = 256;
    kurtosisResult* partials = new kurtosisResult[nGroups];

    #pragma omp target enter data map(to: d_values[0:elemCount]) \
        map(alloc: partials[0:nGroups])

    kurtosisResult result;
    result.count = 0; result.mean = 0; result.m2 = 0; result.m3 = 0; result.m4 = 0;

    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
        // Initialize partials
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int g = 0; g < nGroups; g++) {
            partials[g].count = 0;
            partials[g].mean  = 0;
            partials[g].m2    = 0;
            partials[g].m3    = 0;
            partials[g].m4    = 0;
        }

        // Each group processes a chunk of elements sequentially
        const int chunk = (elemCount + nGroups - 1) / nGroups;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int g = 0; g < nGroups; g++) {
            int start = g * chunk;
            int end   = start + chunk;
            if (end > elemCount) end = elemCount;
            kurtosisResult lres;
            lres.count = 0; lres.mean = 0; lres.m2 = 0; lres.m3 = 0; lres.m4 = 0;
            for (int i = start; i < end; i++) {
                kurtosisResult elem_res;
                elem_res.count = 1; elem_res.mean = d_values[i];
                elem_res.m2 = 0; elem_res.m3 = 0; elem_res.m4 = 0;
                lres = kurtosis_combine(lres, elem_res);
            }
            partials[g] = lres;
        }

        // Combine partials on host
        #pragma omp target update from(partials[0:nGroups])
        result.count = 0; result.mean = 0; result.m2 = 0; result.m3 = 0; result.m4 = 0;
        for (int g = 0; g < nGroups; g++) {
            result = kurtosis_combine(result, partials[g]);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;

    std::cout << "Total device compute time: " << elapsed << " (s)\n";
    std::cout << "Results:\n";
    std::cout << sizeof(kurtosisResult) << " "
              << result.count << " "
              << result.m2    << " "
              << result.m3    << " "
              << result.m4    << "\n";

    #pragma omp target exit data map(delete: d_values[0:elemCount], partials[0:nGroups])

    free(d_values);
    delete[] h_elem;
    delete[] partials;
    return 0;
}

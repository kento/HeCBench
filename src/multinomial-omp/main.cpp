// Multinomial sampling - OpenMP target offloading port
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

template <typename scalar_t, typename accscalar_t>
static void sampleMultinomialOnce_cpu(
    int* dest, int distributions, int categories,
    const scalar_t* sampled, const scalar_t* dist,
    int stride_dist, int stride_categories)
{
    for (int curDist = 0; curDist < distributions; ++curDist) {
        accscalar_t sum = accscalar_t{0};
        for (int cat = 0; cat < categories; ++cat)
            sum += static_cast<accscalar_t>(
                dist[curDist * stride_dist + cat * stride_categories]);
        if (sum == accscalar_t{0}) { dest[curDist] = 0; continue; }

        scalar_t    sample     = sampled[curDist];
        accscalar_t prevBucket = accscalar_t{0};
        accscalar_t curBucket  = accscalar_t{0};
        bool found = false;
        for (int cat = 0; cat < categories; ++cat) {
            accscalar_t dv = static_cast<accscalar_t>(
                dist[curDist * stride_dist + cat * stride_categories]) / sum;
            prevBucket = curBucket; curBucket = prevBucket + dv;
            if (sample < curBucket && sample >= prevBucket && dv > scalar_t{0}) {
                dest[curDist] = cat; found = true; break;
            }
        }
        if (!found) {
            for (int cat = categories - 1; cat >= 0; --cat) {
                if (dist[curDist * stride_dist + cat * stride_categories] > scalar_t{0}) {
                    dest[curDist] = cat; break;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <numDist> <numCategories> <repeat>\n", argv[0]);
        return 1;
    }
    const int numDist       = atoi(argv[1]);
    const int numCategories = atoi(argv[2]);
    const int repeat        = atoi(argv[3]);

    std::vector<float> h_sample(numDist);
    {
        std::default_random_engine rng(123);
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        for (int i = 0; i < numDist; ++i) h_sample[i] = ud(rng);
    }

    int dist_size = numDist * numCategories;
    std::vector<float> h_dist(dist_size);
    {
        srand(123);
        for (int i = 0; i < dist_size; ++i)
            h_dist[i] = static_cast<float>(rand() % 100 + 1);
    }

    std::vector<int> h_result(numDist, 0);
    std::vector<int> h_result_ref(numDist, 0);

    // Allocate and initialise device arrays
    float* d_sample = (float*)malloc(numDist * sizeof(float));
    float* d_dist   = (float*)malloc(dist_size * sizeof(float));
    int*   d_result = (int*)malloc(numDist * sizeof(int));

    for (int i = 0; i < numDist;    ++i) d_sample[i] = h_sample[i];
    for (int i = 0; i < dist_size;  ++i) d_dist[i]   = h_dist[i];

    #pragma omp target enter data map(alloc: d_sample[0:numDist], \
                                             d_dist[0:dist_size], \
                                             d_result[0:numDist])
    #pragma omp target update to(d_sample[0:numDist], d_dist[0:dist_size])

    const int cats = numCategories;

    // Reusable kernel lambda (captures scalars by value, pointers to mapped data)
    auto run_kernel = [=]() {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int curDist = 0; curDist < numDist; curDist++) {
            float sum = 0.f;
            for (int cat = 0; cat < cats; ++cat)
                sum += d_dist[curDist * cats + cat];
            if (sum == 0.f) { d_result[curDist] = 0; continue; }

            float sample     = d_sample[curDist];
            float prevBucket = 0.f, curBucket = 0.f;
            bool  found      = false;
            for (int cat = 0; cat < cats; ++cat) {
                float dv   = d_dist[curDist * cats + cat] / sum;
                prevBucket = curBucket; curBucket = prevBucket + dv;
                if (sample < curBucket && sample >= prevBucket && dv > 0.f) {
                    d_result[curDist] = cat; found = true; break;
                }
            }
            if (!found) {
                for (int cat = cats - 1; cat >= 0; --cat) {
                    if (d_dist[curDist * cats + cat] > 0.f) {
                        d_result[curDist] = cat; break;
                    }
                }
            }
        }
    };

    // Validation run
    run_kernel();

    sampleMultinomialOnce_cpu<float, float>(
        h_result_ref.data(), numDist, numCategories,
        h_sample.data(), h_dist.data(), numCategories, 1);

    #pragma omp target update from(d_result[0:numDist])
    for (int i = 0; i < numDist; ++i) h_result[i] = d_result[i];

    bool error = false;
    for (int i = 0; i < numDist; ++i) {
        if (std::abs(h_result[i] - h_result_ref[i]) > 1) {
            printf("results mismatch: dist=%d omp=%d ref=%d\n",
                   i, h_result[i], h_result_ref[i]);
            error = true; break;
        }
    }
    printf("%s\n", error ? "FAIL" : "PASS");

    // Timed benchmark
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_kernel();
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time of sampleMultinomialOnce kernel: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);

    #pragma omp target exit data map(delete: d_sample[0:numDist], \
                                             d_dist[0:dist_size], \
                                             d_result[0:numDist])
    free(d_sample); free(d_dist); free(d_result);
    return 0;
}

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

// -----------------------------------------------------------------------
// IS benchmark — CLASS S (hardcoded)
// -----------------------------------------------------------------------
#define CLASS            'S'
#define TOTAL_KEYS_LOG_2  16
#define MAX_KEY_LOG_2     11
#define NUM_BUCKETS_LOG_2  9

#define TOTAL_KEYS  (1 << TOTAL_KEYS_LOG_2)   // 65536
#define MAX_KEY     (1 << MAX_KEY_LOG_2)       // 2048
#define NUM_KEYS    TOTAL_KEYS                 // 65536
#define SIZE_OF_BUFFERS NUM_KEYS
#define MAX_ITERATIONS 24
#define TEST_ARRAY_SIZE 5

#define R23 (1.0 / (double)(1<<23))
#define T23 ((double)(1<<23))
#define R46 (R23 * R23)
#define T46 (T23 * T23)

#pragma omp declare target
static double randlc(double *X, const double A) {
  double T1, T2, T3, T4;
  double A1, A2, X1, X2, Z;
  int j;

  T1 = R23 * A;
  j  = (int)T1;
  A1 = j;
  A2 = A - T23 * A1;

  T1 = R23 * *X;
  j  = (int)T1;
  X1 = j;
  X2 = *X - T23 * X1;
  T1 = A1 * X2 + A2 * X1;

  j  = (int)(R23 * T1);
  T2 = j;
  Z  = T1 - T23 * T2;
  T3 = T23 * Z + A2 * X2;
  j  = (int)(R46 * T3);
  T4 = j;
  *X = T3 - T46 * T4;

  return R46 * *X;
}

static double find_my_seed(int kn, int np, long nn, double s, double a) {
  if (kn == 0) return s;
  long mq = (nn / 4 + np - 1) / np;
  long nq = mq * 4 * kn;
  double t1 = s, t2 = a;
  long kk = nq;
  while (kk > 1) {
    long ik = kk / 2;
    if (2 * ik == kk) {
      (void)randlc(&t2, t2);
      kk = ik;
    } else {
      (void)randlc(&t1, t2);
      kk = kk - 1;
    }
  }
  (void)randlc(&t1, t2);
  return t1;
}
#pragma omp end declare target

static void createSeq(int* key_array, double seed, double a, int num_procs) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int myid = 0; myid < num_procs; myid++) {
        double an = a;
        int mq = (NUM_KEYS + num_procs - 1) / num_procs;
        int k1 = mq * myid;
        int k2 = k1 + mq;
        if (k2 > NUM_KEYS) k2 = NUM_KEYS;

        double s = find_my_seed(myid, num_procs, (long)4 * NUM_KEYS, seed, an);
        int k = MAX_KEY / 4;

        for (int i = k1; i < k2; i++) {
            double x = randlc(&s, an);
            x += randlc(&s, an);
            x += randlc(&s, an);
            x += randlc(&s, an);
            key_array[i] = (int)(k * x);
        }
    }
}

int main(int /*argc*/, char ** /*argv*/) {
    static const int S_test_index_array[TEST_ARRAY_SIZE] = {48427,17148,23627,62548,4431};
    static const int S_test_rank_array [TEST_ARRAY_SIZE] = {0,18,346,64917,65463};

    printf("\n\n NAS Parallel Benchmarks 4.1 IS Benchmark\n");
    printf(" Size:  %d  (class %c)\n", (int)TOTAL_KEYS, CLASS);
    printf(" Iterations:   %d\n", MAX_ITERATIONS);

    int* key_array = (int*)malloc(SIZE_OF_BUFFERS * sizeof(int));
    int* key_buff1 = (int*)malloc(MAX_KEY * sizeof(int));
    int* partial_verify_vals = (int*)malloc(TEST_ARRAY_SIZE * sizeof(int));

    #pragma omp target enter data map(alloc: key_array[0:SIZE_OF_BUFFERS], key_buff1[0:MAX_KEY], \
                                             partial_verify_vals[0:TEST_ARRAY_SIZE])

    const int num_create_procs = 64;
    createSeq(key_array, 314159265.0, 1220703125.0, num_create_procs);

    int passed_verification = 0;

    auto t_start = std::chrono::steady_clock::now();

    for (int iter = 1; iter <= MAX_ITERATIONS; iter++) {
        // rank_1: set sentinel keys, save partial-verify vals
        #pragma omp target update from(key_array[0:SIZE_OF_BUFFERS])
        key_array[iter]                  = iter;
        key_array[iter + MAX_ITERATIONS] = MAX_KEY - iter;
        for (int i = 0; i < TEST_ARRAY_SIZE; i++)
            partial_verify_vals[i] = key_array[S_test_index_array[i]];
        #pragma omp target update to(key_array[0:SIZE_OF_BUFFERS], \
                                     partial_verify_vals[0:TEST_ARRAY_SIZE])

        // rank_2: clear histogram
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < MAX_KEY; i++) key_buff1[i] = 0;

        // rank_3: count keys
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < NUM_KEYS; i++) {
            #pragma omp atomic update
            key_buff1[key_array[i]]++;
        }

        // Prefix sum on host
        #pragma omp target update from(key_buff1[0:MAX_KEY])
        int sum = 0;
        for (int i = 0; i < MAX_KEY; i++) {
            sum += key_buff1[i];
            key_buff1[i] = sum;
        }
        #pragma omp target update to(key_buff1[0:MAX_KEY])

        // Partial verification
        for (int i = 0; i < TEST_ARRAY_SIZE; i++) {
            int k = partial_verify_vals[i];
            if (k > 0 && k <= NUM_KEYS - 1) {
                int key_rank = key_buff1[k - 1];
                int failed   = 0;
                if (i <= 2) {
                    if (key_rank != S_test_rank_array[i] + iter) failed = 1;
                    else passed_verification++;
                } else {
                    if (key_rank != S_test_rank_array[i] - iter) failed = 1;
                    else passed_verification++;
                }
                if (failed)
                    printf("Failed partial verification: iteration %d, test key %d\n",
                           iter, i);
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    printf("Average execution time of the rank kernels %f (s)\n",
           total_s / MAX_ITERATIONS);

    // Full verify
    #pragma omp target update from(key_array[0:SIZE_OF_BUFFERS])
    std::vector<int> hist(MAX_KEY, 0);
    for (int i = 0; i < NUM_KEYS; i++) hist[key_array[i]]++;
    int sv = 0;
    for (int i = 0; i < MAX_KEY; i++) { sv += hist[i]; hist[i] = sv; }

    std::vector<int> sorted(NUM_KEYS);
    std::vector<int> inv_count(MAX_KEY, 0);
    std::vector<int> start_pos(MAX_KEY, 0);
    start_pos[0] = 0;
    for (int i = 1; i < MAX_KEY; i++) start_pos[i] = hist[i - 1];
    for (int i = 0; i < NUM_KEYS; i++) {
        int v = key_array[i];
        sorted[start_pos[v] + inv_count[v]] = v;
        inv_count[v]++;
    }

    int out_of_order = 0;
    for (int i = 1; i < NUM_KEYS; i++)
        if (sorted[i - 1] > sorted[i]) out_of_order++;

    if (out_of_order != 0)
        printf("Full_verify: number of keys out of sort: %d\n", out_of_order);
    else
        passed_verification++;

    bool pass = (passed_verification == 5 * MAX_ITERATIONS + 1);
    printf("%s\n", pass ? "PASS" : "FAIL");

    #pragma omp target exit data map(delete: key_array[0:SIZE_OF_BUFFERS], \
        key_buff1[0:MAX_KEY], partial_verify_vals[0:TEST_ARRAY_SIZE])
    free(key_array); free(key_buff1); free(partial_verify_vals);
    return 0;
}

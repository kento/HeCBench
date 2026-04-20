// OpenMP target offloading port of streamUM benchmark
// Task consumer executing gemv operations

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
void srand48(long seed) { srand((unsigned int)seed); }
double drand48() { return double(rand()) / RAND_MAX; }
#endif

template <typename T>
static void gemv_host(int m, int n, T alpha, const T *A, const T *x, T beta, T *result) {
  for (int i = 0; i < m; i++) {
    T s = T(0);
    for (int j = 0; j < n; j++)
      s += A[i * n + j] * x[j];
    result[i] = alpha * s + beta * result[i];
  }
}

template <typename T>
struct Task {
  unsigned int size, id;
  std::vector<T> data;
  std::vector<T> result;
  std::vector<T> vec;

  Task() : size(0), id(0) {}

  void allocate(unsigned int s, unsigned int uid) {
    id = uid; size = s;
    data.resize((size_t)s * s);
    result.resize(s, T(0));
    vec.resize(s);
    for (size_t i = 0; i < data.size(); i++) data[i] = (T)drand48();
    for (size_t i = 0; i < vec.size(); i++) { result[i] = T(0); vec[i] = (T)drand48(); }
  }
};

template <typename T>
static void execute_task(Task<T> &t) {
  if (t.size < 100) {
    gemv_host<T>(t.size, t.size, T(1.0), t.data.data(), t.vec.data(), T(0.0), t.result.data());
  } else {
    const int M = (int)t.size;
    const int N = (int)t.size;
    T *A   = t.data.data();
    T *x   = t.vec.data();
    T *r   = t.result.data();

    #pragma omp target data map(to: A[0:M*N], x[0:N]) map(tofrom: r[0:M])
    {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < M; i++) {
        T s = T(0);
        for (int j = 0; j < N; j++)
          s += A[i * N + j] * x[j];
        r[i] = s;
      }
    }
  }
}

template <typename T>
static void check(const std::vector<Task<T>> &tasks) {
  bool ok = true;
  for (const auto &t : tasks) {
    if (t.size < 100) continue;
    std::vector<T> ref(t.size, T(0));
    for (int i = 0; i < (int)t.size; i++)
      for (int j = 0; j < (int)t.size; j++)
        ref[i] += t.data[j * t.size + i] * t.vec[j];
    for (int j = 0; j < (int)t.size; j++) {
      if (std::fabs(t.result[j] - ref[j]) > 1e-3) { ok = false; break; }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <nthreads> <ntasks> <verify>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[2]);
  const int verify = atoi(argv[3]);

  srand48(48);
  std::vector<Task<double>> tasks(N);
  for (int i = 0; i < N; i++) {
    int s = std::max((int)(drand48() * 1000.0), 64);
    tasks[i].allocate(s, i);
  }

  printf("Executing tasks on device\n");
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < N; i++)
    execute_task(tasks[i]);
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Task execution time : %f (s)\n", time * 1e-9f);

  if (verify) check(tasks);
  return 0;
}

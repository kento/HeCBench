// OpenMP target offloading port of streamCreateCopyDestroy benchmark
// Measures overhead of repeated target data region creation/destruction

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#define BufSize    0x1000
#define Iterations 0x100
#define TotalStreams 4
#define TotalBufs    4

static const size_t totalStreams[TotalStreams] = {1, 2, 4, 8};
static const size_t totalBuffers[TotalBufs]   = {1, 100, 1000, 5000};

// Baseline: just measure allocation/copy time without "stream" overhead
static void run_baseline(unsigned int testNumber) {
  size_t numBuffers = totalBuffers[testNumber / TotalBufs];
  size_t nBytes = BufSize * sizeof(float);

  std::vector<float> hSrc(BufSize);
  for (size_t i = 0; i < BufSize; i++) hSrc[i] = 1.618f + (float)i;

  std::vector<std::vector<float>> bufs(numBuffers, std::vector<float>(BufSize));

  auto start = std::chrono::steady_clock::now();

  for (size_t b = 0; b < numBuffers; ++b) {
    float *h = hSrc.data();
    float *d = bufs[b].data();
    #pragma omp target data map(to: h[0:BufSize]) map(alloc: d[0:BufSize])
    {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < BufSize; i++) d[i] = h[i];
    }
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
  std::cout << std::setw(6) << numBuffers << " buffers baseline: "
            << std::fixed << std::setprecision(3) << ms << " ms\n";
}

// Stream test: measures repeated target region creation (simulates stream create/destroy)
static void run_stream(unsigned int testNumber) {
  size_t numStreams = totalStreams[testNumber % TotalStreams];
  size_t numBuffers = totalBuffers[testNumber / TotalBufs];
  size_t iter = Iterations / (numStreams * ((size_t)1 << (testNumber / TotalBufs + 1)));
  if (iter == 0) iter = 1;

  size_t nBytes = BufSize * sizeof(float);
  std::vector<float> hSrc(BufSize);
  for (size_t i = 0; i < BufSize; i++) hSrc[i] = 1.618f + (float)i;

  std::vector<std::vector<float>> device_bufs(numBuffers, std::vector<float>(BufSize, 0.f));
  float *h = hSrc.data();

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    // Each "stream" is simulated by a separate target data region + update
    for (size_t s = 0; s < numStreams; ++s) {
      for (size_t b = 0; b < numBuffers; ++b) {
        float *d = device_bufs[b].data();
        #pragma omp target data map(to: h[0:BufSize]) map(tofrom: d[0:BufSize])
        {
          #pragma omp target teams distribute parallel for thread_limit(256)
          for (int k = 0; k < BufSize; k++) d[k] = h[k];
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
  std::cout << std::setw(2) << numStreams << " streams, "
            << std::setw(6) << numBuffers << " buffers: "
            << std::fixed << std::setprecision(3) << ms << " ms"
            << "  (" << iter << " iterations)\n";
}

int main(int argc, char **argv) {
  std::cout << "=== Stream Create Copy Destroy Benchmark (OpenMP) ===\n\n";

  std::cout << "--- Baseline (no stream overhead) ---\n";
  for (unsigned int t = 0; t < TotalBufs; ++t)
    run_baseline(t * TotalStreams);

  std::cout << "\n--- Stream tests ---\n";
  for (unsigned int t = 0; t < TotalStreams * TotalBufs; ++t)
    run_stream(t);

  return 0;
}

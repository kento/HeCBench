#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <cstring>

void preprocessing(float* data_t, int N, int L) {
  for (int i = 0; i < N; i++) {
    float* row = data_t + i * L;
    double sum1 = 0.0, sum2 = 0.0;
    for (int l = 0; l < L; l++) sum1 += row[l];
    sum1 /= L;
    for (int l = 0; l < L; l++) sum2 += (row[l] - sum1) * (row[l] - sum1);
    sum2 = std::sqrt(sum2);
    for (int l = 0; l < L; l++) {
      if (sum2 != 0.0) row[l] = (float)((row[l] - sum1) / sum2);
      else row[l] = 0.f;
    }
  }
}

int CorMat_singlePass(float* upper_tri, float* data, int N, int L) {
  const size_t M1    = (size_t)(N - 1) * N / 2;
  const size_t total = (size_t)N * N;

  preprocessing(data, N, L);

  // Allocate device-only cormat; data and upper_tri are mapped directly
  float* d_cormat = (float*)malloc((size_t)N * N * sizeof(float));

#pragma omp target enter data map(to: data[0:N*L]) \
    map(alloc: d_cormat[0:N*N], upper_tri[0:M1])

  auto t0 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < L; k++) sum += data[i * L + k] * data[j * L + k];
      d_cormat[i * N + j] = sum;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  long long gemm_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  t0 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < (int)total; idx++) {
    int i = idx / N, j = idx % N;
    if (i < j) {
      size_t t = (size_t)N * i - (size_t)i * (i + 1) / 2 + j - i - 1;
      upper_tri[t] = d_cormat[i * N + j];
    }
  }
  t1 = std::chrono::steady_clock::now();
  long long extract_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

#pragma omp target update from(upper_tri[0:M1])
#pragma omp target exit data map(delete: data[0:N*L], d_cormat[0:N*N], upper_tri[0:M1])

  free(d_cormat);

  std::cout << "Kernel time (s)\nGEMM: " << gemm_ns * 1e-9 << ", "
            << "Extract upper triangle: " << extract_ns * 1e-9 << "\n";
  return 1;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <Number of voxels> <Length of time series>\n";
    return 1;
  }
  int N = atoi(argv[1]), L = atoi(argv[2]);
  std::cout << "Number of voxels: " << N << "  Length of time series: " << L << "\n\n";

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-6.f, 6.f);
  float* data = new float[N * L];
  for (int k = 0; k < N; k++)
    for (int l = 0; l < L; l++)
      data[k * L + l] = distr(g);

  size_t M11 = (size_t)(N - 1) * N / 2;
  float* upper_tri = new float[M11]();

  std::cout << "\nComputing correlations ...\n";
  {
    auto t0 = std::chrono::steady_clock::now();
    CorMat_singlePass(upper_tri, data, N, L);
    auto t1 = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::cout << "\nRunning time for computing correlations:\n" << (time * 1e-9f) << " (s)\n";

    double checksum = 0.0;
    for (size_t i = 0; i < M11; i++) checksum += upper_tri[i];
    std::cout << "Checksum: " << checksum << "\n";

    if (N < 100 && L < 100) {
      std::cout << "\nWriting correlation values into corrs.txt ...\n";
      std::ofstream f("corrs.txt");
      for (size_t i = 0; i < M11; i++) f << upper_tri[i] << '\n';
      std::cout << "Correlations stored in corrs.txt\n";
    }
  }
  delete[] upper_tri;
  delete[] data;
  return 0;
}

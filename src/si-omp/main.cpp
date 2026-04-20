// Slit diffraction via 2D FFT (OpenMP target port of si-kokkos)
// Implements Cooley-Tukey radix-2 FFT applied row-wise then column-wise.

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Complex {
  double re, im;
};

// In-place 1D Cooley-Tukey radix-2 FFT (forward, n must be power of 2)
#pragma omp declare target
static void fft1d(Complex* data, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      Complex tmp = data[i];
      data[i] = data[j];
      data[j] = tmp;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = -2.0 * M_PI / len;
    double wlen_re = cos(ang), wlen_im = sin(ang);
    for (int i = 0; i < n; i += len) {
      double w_re = 1.0, w_im = 0.0;
      for (int jj = 0; jj < len / 2; jj++) {
        Complex u = data[i + jj];
        Complex v;
        v.re = data[i + jj + len/2].re * w_re - data[i + jj + len/2].im * w_im;
        v.im = data[i + jj + len/2].re * w_im + data[i + jj + len/2].im * w_re;
        data[i + jj].re           = u.re + v.re;
        data[i + jj].im           = u.im + v.im;
        data[i + jj + len/2].re   = u.re - v.re;
        data[i + jj + len/2].im   = u.im - v.im;
        double new_w_re = w_re * wlen_re - w_im * wlen_im;
        double new_w_im = w_re * wlen_im + w_im * wlen_re;
        w_re = new_w_re; w_im = new_w_im;
      }
    }
  }
}
#pragma omp end declare target

// Reference direct 2D DFT (O(N^4), CPU-only, for small N verification)
void reference_dft(const double* input, double* output, int N) {
  for (int k = 0; k < N; k++) {
    for (int l = 0; l < N; l++) {
      double re = 0.0, im = 0.0;
      for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
          double angle = -2.0 * M_PI * ((double)(j * k + i * l)) / N;
          re += input[j * N + i] * cos(angle);
          im += input[j * N + i] * sin(angle);
        }
      }
      output[k * N + l] = re * re + im * im;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <transform_size_N> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  if (N <= 0 || (N & (N - 1)) != 0) {
    printf("Error: N must be a positive power of 2\n");
    return 1;
  }

  printf("Running FFT for %d x %d = %d = 2 ^ %d data points...\n",
         N, N, N * N, (int)(log2((double)(N * N))));

  long long nn = (long long)N * N;
  Complex *d      = (Complex*)malloc(nn * sizeof(Complex));
  Complex *scratch = (Complex*)malloc(nn * sizeof(Complex));

  std::vector<double> input_ref(nn, 0.0);

  const int slit_height = 4;
  const int slit_width  = 2;
  const int slit_dist   = 8;

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      int di = (i - N/2); if (di < 0) di = -di;
      int dj = (j - N/2); if (dj < 0) dj = -dj;
      if (di >= slit_dist && di <= slit_dist + slit_width && dj <= slit_height) {
        d[j*N+i].re = 1.0; d[j*N+i].im = 0.0;
        input_ref[j*N+i] = 1.0;
      } else {
        d[j*N+i].re = 0.0; d[j*N+i].im = 0.0;
      }
    }
  }

  // Save initial data
  Complex *d_init = (Complex*)malloc(nn * sizeof(Complex));
  memcpy(d_init, d, nn * sizeof(Complex));

  #pragma omp target enter data map(alloc: d[0:nn], scratch[0:nn])

  // Warmup
  {
    memcpy(d, d_init, nn * sizeof(Complex));
    #pragma omp target update to(d[0:nn])
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int row = 0; row < N; row++) fft1d(&d[row*N], N);
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int col = 0; col < N; col++) {
      for (int k = 0; k < N; k++) scratch[col*N+k] = d[k*N+col];
      fft1d(&scratch[col*N], N);
      for (int k = 0; k < N; k++) d[k*N+col] = scratch[col*N+k];
    }
  }

  double total_ms = 0.0;
  for (int r = 0; r < repeat; r++) {
    memcpy(d, d_init, nn * sizeof(Complex));
    #pragma omp target update to(d[0:nn])

    auto t0 = std::chrono::steady_clock::now();

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int row = 0; row < N; row++) fft1d(&d[row*N], N);

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int col = 0; col < N; col++) {
      for (int k = 0; k < N; k++) scratch[col*N+k] = d[k*N+col];
      fft1d(&scratch[col*N], N);
      for (int k = 0; k < N; k++) d[k*N+col] = scratch[col*N+k];
    }

    auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6;
  }

  printf("Average execution time of FFT: %lf ms\n", total_ms / repeat);

  #pragma omp target update from(d[0:nn])
  #pragma omp target exit data map(delete: d[0:nn], scratch[0:nn])

  std::vector<double> output(nn);
  for (long long idx = 0; idx < nn; idx++) {
    output[idx] = d[idx].re * d[idx].re + d[idx].im * d[idx].im;
  }

  bool ok = true;
  if (N <= 64) {
    std::vector<double> ref_output(nn, 0.0);
    reference_dft(input_ref.data(), ref_output.data(), N);
    for (long long i = 0; i < nn; i++) {
      double ref = ref_output[i];
      double tol = 1e-3 * (1.0 + ref);
      if (std::abs(output[i] - ref) > tol) { ok = false; break; }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(d); free(scratch); free(d_init);
  return 0;
}

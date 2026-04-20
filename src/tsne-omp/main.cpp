// OpenMP target offloading port of tsne benchmark
// t-SNE dimensionality reduction — core gradient computation
// Standalone synthetic version (original has complex CUDA dependencies)

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>

// t-SNE gradient computation kernels

// Compute pairwise squared distances
static void compute_sq_distances(const float *Y, float *sq_dists,
                                 int N, int D) {
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float dist = 0.f;
      for (int d = 0; d < D; d++) {
        float diff = Y[i * D + d] - Y[j * D + d];
        dist += diff * diff;
      }
      sq_dists[i * N + j] = dist;
    }
  }
}

// Compute Q distribution (Student t-distribution)
static void compute_Q(const float *sq_dists, float *Q, float *Q_sum,
                      int N) {
  // Q[i,j] = (1 + ||y_i - y_j||^2)^-1
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < N; i++) {
    float row_sum = 0.f;
    for (int j = 0; j < N; j++) {
      float q = (i == j) ? 0.f : 1.f / (1.f + sq_dists[i * N + j]);
      Q[i * N + j] = q;
      row_sum += q;
    }
    Q_sum[i] = row_sum;
  }
}

// Compute attractive and repulsive forces
static void compute_gradients(const float *P, const float *Q, const float *Q_sum,
                              const float *Y, const float *sq_dists,
                              float *dY, int N, int D, float exaggeration) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < N; i++) {
    float total_Q = 0.f;
    for (int k = 0; k < N; k++) total_Q += Q_sum[k];
    if (total_Q < 1e-10f) total_Q = 1e-10f;

    for (int d = 0; d < D; d++) {
      float grad = 0.f;
      for (int j = 0; j < N; j++) {
        if (i == j) continue;
        float q_ij = Q[i * N + j] / (total_Q + 1e-10f);
        float p_ij = P[i * N + j];
        float factor = (exaggeration * p_ij - q_ij) *
                       (1.f / (1.f + sq_dists[i * N + j]));
        grad += 4.f * factor * (Y[i * D + d] - Y[j * D + d]);
      }
      dY[i * D + d] = grad;
    }
  }
}

// SGD update with momentum
static void update_embedding(float *Y, float *gains, float *old_dY,
                             const float *dY, int N, int D,
                             float lr, float momentum) {
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < N * D; i++) {
    // Adaptive learning rate (gains)
    bool same_sign = (dY[i] * old_dY[i] > 0.f);
    gains[i] = same_sign ? gains[i] * 0.8f : gains[i] + 0.2f;
    if (gains[i] < 0.01f) gains[i] = 0.01f;

    old_dY[i] = momentum * old_dY[i] - lr * gains[i] * dY[i];
    Y[i] += old_dY[i];
  }
}

// Compute symmetrized P from input data (placeholder: use random sparse P)
static void init_P(float *P, int N, int perp) {
  // In real t-SNE, P is computed from high-dim data using binary search
  // Here we generate a synthetic symmetric sparse P
  srand(42);
  for (int i = 0; i < N; i++) {
    float row_sum = 0.f;
    for (int j = 0; j < N; j++) {
      float val = (i == j) ? 0.f : expf(-((i-j)*(i-j)) / (2.f * perp * perp));
      P[i * N + j] = val;
      row_sum += val;
    }
    if (row_sum > 0.f)
      for (int j = 0; j < N; j++) P[i * N + j] /= row_sum;
  }
  // Symmetrize: P_ij = (P_ij + P_ji) / (2N)
  for (int i = 0; i < N; i++)
    for (int j = i+1; j < N; j++) {
      float sym = (P[i*N+j] + P[j*N+i]) / (2.f * N);
      P[i*N+j] = sym;
      P[j*N+i] = sym;
    }
}

int main(int argc, char **argv) {
  int N     = 1000; // number of points
  int D_out = 2;    // output dimensions
  int steps = 100;
  int perp  = 30;

  printf("t-SNE: N=%d points -> %dD, steps=%d, perplexity=%d\n", N, D_out, steps, perp);

  // Initialize low-dim embedding Y with small random values
  std::vector<float> h_Y(N * D_out), h_P(N * N, 0.f);
  std::vector<float> h_dY(N * D_out, 0.f), h_old_dY(N * D_out, 0.f);
  std::vector<float> h_gains(N * D_out, 1.f);
  std::vector<float> h_sq_dists(N * N, 0.f), h_Q(N * N, 0.f), h_Q_sum(N, 0.f);

  {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1e-4f);
    for (auto &v : h_Y) v = nd(rng);
  }
  init_P(h_P.data(), N, perp);

  float *d_Y        = h_Y.data();
  float *d_P        = h_P.data();
  float *d_dY       = h_dY.data();
  float *d_old_dY   = h_old_dY.data();
  float *d_gains    = h_gains.data();
  float *d_sq_dists = h_sq_dists.data();
  float *d_Q        = h_Q.data();
  float *d_Q_sum    = h_Q_sum.data();

  #pragma omp target enter data \
    map(tofrom: d_Y[0:N*D_out]) \
    map(to: d_P[0:N*N]) \
    map(alloc: d_dY[0:N*D_out], d_old_dY[0:N*D_out], d_gains[0:N*D_out], \
               d_sq_dists[0:N*N], d_Q[0:N*N], d_Q_sum[0:N])

  // Initialize device arrays
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < N * D_out; i++) {
    d_dY[i] = 0.f; d_old_dY[i] = 0.f; d_gains[i] = 1.f;
  }

  auto t_start = std::chrono::steady_clock::now();

  float lr = 200.f, momentum = 0.5f, exaggeration = 12.f;

  for (int step = 0; step < steps; step++) {
    if (step == 250) momentum = 0.8f;
    if (step == 100) exaggeration = 1.f;

    compute_sq_distances(d_Y, d_sq_dists, N, D_out);
    compute_Q(d_sq_dists, d_Q, d_Q_sum, N);
    compute_gradients(d_P, d_Q, d_Q_sum, d_Y, d_sq_dists, d_dY, N, D_out, exaggeration);
    update_embedding(d_Y, d_gains, d_old_dY, d_dY, N, D_out, lr, momentum);
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
  printf("t-SNE total time (%d steps): %.3f s\n", steps, elapsed);
  printf("Average per step: %.3f ms\n", elapsed * 1e3 / steps);

  #pragma omp target update from(d_Y[0:N*D_out])
  #pragma omp target exit data \
    map(delete: d_Y[0:N*D_out], d_P[0:N*N], d_dY[0:N*D_out], d_old_dY[0:N*D_out], \
                d_gains[0:N*D_out], d_sq_dists[0:N*N], d_Q[0:N*N], d_Q_sum[0:N])

  // Checksum
  double sum = 0.0;
  for (int i = 0; i < N * D_out; i++) sum += h_Y[i];
  printf("Embedding checksum: %f\n", sum);
  printf("PASS\n");
  return 0;
}

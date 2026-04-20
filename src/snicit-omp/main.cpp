// SNICIT OpenMP target port: simplified sparse neural network inference
// Weight matrices stored in CSC format; batch inference with ReLU activations.

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <sys/stat.h>

struct LayerCSC {
  int   *col_ptr;   // [num_neurons + 1]
  int   *row_idx;   // [nnz]
  float *vals;      // [nnz]
  float *bias;      // [num_neurons]
  int    num_neurons;
  int    nnz;
};

LayerCSC make_synthetic_layer(int num_neurons, float density, std::mt19937& rng) {
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::normal_distribution<float>       gauss(0.0f, 1.0f);

  std::vector<int>   h_col_ptr;
  std::vector<int>   h_row_idx;
  std::vector<float> h_vals;
  std::vector<float> h_bias(num_neurons);

  h_col_ptr.reserve(num_neurons + 1);
  h_col_ptr.push_back(0);

  for (int j = 0; j < num_neurons; j++) {
    for (int i = 0; i < num_neurons; i++) {
      if (uniform(rng) < density) {
        h_row_idx.push_back(i);
        h_vals.push_back(gauss(rng) * 0.1f);
      }
    }
    h_col_ptr.push_back((int)h_row_idx.size());
    h_bias[j] = gauss(rng) * 0.01f;
  }

  int nnz = (int)h_row_idx.size();
  LayerCSC layer;
  layer.num_neurons = num_neurons;
  layer.nnz = nnz;
  layer.col_ptr = (int*)malloc((num_neurons+1)*sizeof(int));
  layer.row_idx = (int*)malloc(nnz*sizeof(int));
  layer.vals    = (float*)malloc(nnz*sizeof(float));
  layer.bias    = (float*)malloc(num_neurons*sizeof(float));

  memcpy(layer.col_ptr, h_col_ptr.data(), (num_neurons+1)*sizeof(int));
  memcpy(layer.row_idx, h_row_idx.data(), nnz*sizeof(int));
  memcpy(layer.vals,    h_vals.data(),    nnz*sizeof(float));
  memcpy(layer.bias,    h_bias.data(),    num_neurons*sizeof(float));

  return layer;
}

double run_inference(
    const std::vector<LayerCSC>& layers,
    float *buf0,
    float *buf1,
    int batch_size,
    int num_neurons,
    int num_layers)
{
  float *cur = buf0;
  float *nxt = buf1;

  auto t0 = std::chrono::steady_clock::now();

  for (int layer = 0; layer < num_layers; layer++) {
    const int   *col_ptr = layers[layer].col_ptr;
    const int   *row_idx = layers[layer].row_idx;
    const float *vals    = layers[layer].vals;
    const float *bias    = layers[layer].bias;
    const int    n       = layers[layer].num_neurons;

    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
      map(to: col_ptr[0:n+1], row_idx[0:layers[layer].nnz], \
              vals[0:layers[layer].nnz], bias[0:n]) \
      map(tofrom: cur[0:batch_size*n]) \
      map(from: nxt[0:batch_size*n])
    for (int b = 0; b < batch_size; b++) {
      for (int j = 0; j < n; j++) {
        float sum = bias[j];
        int start = col_ptr[j];
        int end   = col_ptr[j + 1];
        for (int k = start; k < end; k++) {
          sum += vals[k] * cur[b * n + row_idx[k]];
        }
        nxt[b * n + j] = sum > 0.0f ? sum : 0.0f;
      }
    }

    float *tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6;
}

static std::string get_opt(int argc, char* argv[], const char* flag, const char* dflt) {
  for (int i = 1; i + 1 < argc; i++)
    if (strcmp(argv[i], flag) == 0) return argv[i + 1];
  return dflt;
}
static int get_opt_int(int argc, char* argv[], const char* flag, int dflt) {
  for (int i = 1; i + 1 < argc; i++)
    if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
  return dflt;
}

int main(int argc, char* argv[]) {
  std::string benchmark  = get_opt(argc, argv, "-k", "A");
  std::string path       = get_opt(argc, argv, "-p", "../dataset");
  int         num_input  = get_opt_int(argc, argv, "-n", 10000);
  int         batch_size = get_opt_int(argc, argv, "-b", 10000);

  int   num_hidden_neurons;
  int   num_layers;
  float density;

  if      (benchmark == "A") { num_hidden_neurons = 128; num_layers = 18; density = 0.6f; }
  else if (benchmark == "B") { num_hidden_neurons = 256; num_layers = 18; density = 0.6f; }
  else if (benchmark == "C") { num_hidden_neurons = 256; num_layers = 12; density = 0.5f; }
  else if (benchmark == "D") { num_hidden_neurons = 256; num_layers = 12; density = 0.5f; }
  else {
    fprintf(stderr, "Unknown benchmark '%s'. Use A, B, C, or D.\n", benchmark.c_str());
    return 1;
  }

  struct stat st;
  bool use_synthetic = (stat(path.c_str(), &st) != 0);
  if (use_synthetic)
    printf("[Warning] Dataset path '%s' not found – using synthetic data.\n\n", path.c_str());

  printf("Benchmark: %s\n",          benchmark.c_str());
  printf("num_hidden_neurons = %d\n", num_hidden_neurons);
  printf("num_layers = %d\n",         num_layers);
  printf("num_inputs = %d\n",         num_input);
  printf("batch_size = %d\n",         batch_size);

  if (batch_size > num_input) batch_size = num_input;
  int num_batches = (num_input + batch_size - 1) / batch_size;

  std::mt19937 rng(12345);

  printf("Generating synthetic weight matrices...\n");
  std::vector<LayerCSC> layers;
  layers.reserve(num_layers);
  for (int l = 0; l < num_layers; l++)
    layers.push_back(make_synthetic_layer(num_hidden_neurons, density, rng));

  size_t buf_size = (size_t)batch_size * num_hidden_neurons;
  float *buf0 = (float*)malloc(buf_size * sizeof(float));
  float *buf1 = (float*)malloc(buf_size * sizeof(float));

  {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (size_t i = 0; i < buf_size; i++)
      buf0[i] = u(rng) > 0.5f ? 1.0f : 0.0f;
  }

  // Warmup
  run_inference(layers, buf0, buf1, batch_size, num_hidden_neurons, num_layers);

  printf("Running inference...\n");
  double total_ms = 0.0;
  for (int bat = 0; bat < num_batches; bat++)
    total_ms += run_inference(layers, buf0, buf1, batch_size, num_hidden_neurons, num_layers);

  double avg_ms = total_ms / num_batches;
  double thpt   = 1000.0 / avg_ms;

  printf("Average inference time per batch: %lf ms\n", avg_ms);
  printf("Throughput: %lf batches/sec\n", thpt);

  free(buf0); free(buf1);
  for (auto& l : layers) {
    free(l.col_ptr); free(l.row_idx); free(l.vals); free(l.bias);
  }
  return 0;
}

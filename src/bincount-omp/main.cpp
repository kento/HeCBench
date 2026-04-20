/*
 * OpenMP target offloading port of bincount benchmark.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <omp.h>

#pragma omp declare target
template <typename input_t, typename output_t, typename IndexType>
IndexType getBin(input_t v, input_t minvalue, input_t maxvalue, IndexType nbins) {
  IndexType bin = (IndexType)((v - minvalue) * nbins / (maxvalue - minvalue));
  if (bin == nbins) bin--;
  return bin;
}
#pragma omp end declare target

template <typename output_t, typename input_t, typename IndexType>
void bincount_global(
    output_t* output,
    const input_t* input,
    IndexType nbins, input_t minvalue, input_t maxvalue, IndexType input_size,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (IndexType i = 0; i < input_size; i++) {
      input_t v = input[i];
      if (v >= minvalue && v <= maxvalue) {
        IndexType bin = getBin(v, minvalue, maxvalue, nbins);
        #pragma omp atomic update
        output[bin]++;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of bincount kernel (global): %f (us)\n",
         (time * 1e-3f) / repeat);
}

template <typename output_t, typename input_t, typename IndexType>
void bincount_shared(
    output_t* output,
    const input_t* input,
    IndexType nbins, input_t minvalue, input_t maxvalue, IndexType input_size,
    int repeat)
{
  const int block_size = 256;
  int nteams = (input_size + block_size - 1) / block_size;

  // Allocate per-block scratch on device: nteams * nbins
  output_t* scratch = (output_t*)calloc((size_t)nteams * nbins, sizeof(output_t));
  #pragma omp target enter data map(alloc: scratch[0:(size_t)nteams*nbins])

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    // Init scratch
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < nteams * nbins; i++) scratch[i] = 0;

    // Accumulate in scratch (one team per block)
    #pragma omp target teams distribute num_teams(nteams) thread_limit(block_size)
    for (int bid = 0; bid < nteams; bid++) {
      #pragma omp parallel for
      for (int t = 0; t < block_size; t++) {
        IndexType li = (IndexType)(bid * block_size + t);
        if (li < input_size) {
          input_t v = input[li];
          if (v >= minvalue && v <= maxvalue) {
            IndexType bin = getBin(v, minvalue, maxvalue, nbins);
            #pragma omp atomic update
            scratch[(size_t)bid * nbins + bin]++;
          }
        }
      }
    }

    // Flush scratch to output
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < nteams * nbins; i++) {
      IndexType bin = i % nbins;
      #pragma omp atomic update
      output[bin] += scratch[i];
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of bincount kernel (shared+global): %f (us)\n",
         (time * 1e-3f) / repeat);

  #pragma omp target exit data map(delete: scratch[0:(size_t)nteams*nbins])
  free(scratch);
}

template <typename output_t, typename input_t, typename IndexType>
void eval(IndexType input_size, int repeat) {
  std::vector<input_t> h_input(input_size);
  std::default_random_engine gen(123);
  std::normal_distribution<input_t> dist(5.0, 2.0);
  for (auto& v : h_input) v = dist(gen);

  input_t minval = *std::min_element(h_input.begin(), h_input.end());
  input_t maxval = *std::max_element(h_input.begin(), h_input.end());
  printf("Input min, max values: (%f %f)\n", (float)minval, (float)maxval);

  input_t* d_input = (input_t*)malloc(input_size * sizeof(input_t));
  for (IndexType i = 0; i < input_size; i++) d_input[i] = h_input[i];
  #pragma omp target enter data map(to: d_input[0:input_size])

  for (IndexType nbins = 768; nbins <= 768 * 32; nbins *= 2) {
    printf("\nNumber of bins: %d\n", (int)nbins);

    output_t* d_output = (output_t*)calloc(nbins, sizeof(output_t));
    #pragma omp target enter data map(alloc: d_output[0:nbins])

    printf("bincount using global atomics\n");
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (IndexType i = 0; i < nbins; i++) d_output[i] = 0;
    bincount_global<output_t, input_t, IndexType>(
        d_output, d_input, nbins, minval, maxval, input_size, repeat);

    #pragma omp target update from(d_output[0:nbins])
    output_t omin = *std::min_element(d_output, d_output+nbins);
    output_t omax = *std::max_element(d_output, d_output+nbins);
    printf("Output min, median, max values: (%ld %ld %ld)\n",
           (int64_t)omin/repeat, (int64_t)d_output[nbins/2]/repeat, (int64_t)omax/repeat);

    printf("\nbincount using global and local atomics\n");
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (IndexType i = 0; i < nbins; i++) d_output[i] = 0;
    bincount_shared<output_t, input_t, IndexType>(
        d_output, d_input, nbins, minval, maxval, input_size, repeat);
    #pragma omp target update from(d_output[0:nbins])
    omin = *std::min_element(d_output, d_output+nbins);
    omax = *std::max_element(d_output, d_output+nbins);
    printf("Output min, median, max values: (%ld %ld %ld)\n\n",
           (int64_t)omin/repeat, (int64_t)d_output[nbins/2]/repeat, (int64_t)omax/repeat);

    #pragma omp target exit data map(delete: d_output[0:nbins])
    free(d_output);
  }

  #pragma omp target exit data map(delete: d_input[0:input_size])
  free(d_input);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  eval<int, float, int>(n, repeat);
  return 0;
}

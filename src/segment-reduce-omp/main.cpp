// OpenMP target port of segment-reduce benchmark.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define VALUE 1

static void segreduce(const size_t num_elements, const int repeat)
{
  printf("num_elements = %zu\n", num_elements);

  int* d_in = (int*) malloc(num_elements * sizeof(int));
  for (size_t i = 0; i < num_elements; i++) d_in[i] = VALUE;

  #pragma omp target enter data map(to: d_in[0:num_elements])

  for (size_t segment_size = 16; segment_size <= 16384; segment_size *= 2) {
    const size_t num_segments = num_elements / segment_size;

    int* d_out = (int*) malloc(num_segments * sizeof(int));
    #pragma omp target enter data map(alloc: d_out[0:num_segments])

    const size_t seg   = segment_size;
    const size_t nsegs = num_segments;

    // Warmup
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int s = 0; s < (int)nsegs; s++) {
      int sum = 0;
      for (size_t i = 0; i < seg; i++) sum += d_in[(size_t)s * seg + i];
      d_out[s] = sum;
    }

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int s = 0; s < (int)nsegs; s++) {
        int sum = 0;
        for (size_t i = 0; i < seg; i++) sum += d_in[(size_t)s * seg + i];
        d_out[s] = sum;
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    printf("num_segments = %zu segment_size = %zu Throughput = %f (G/s)\n",
           num_segments, segment_size,
           1.0 * num_elements * repeat / ns);

    #pragma omp target update from(d_out[0:num_segments])
    #pragma omp target exit data map(delete: d_out[0:num_segments])

    int expected = (int)segment_size * VALUE;
    int errors = 0;
    for (size_t i = 0; i < num_segments; i++) {
      if (d_out[i] != expected) {
        errors++;
        if (errors < 10)
          printf("segment %zu has sum %d (expected %d)\n", i, d_out[i], expected);
      }
    }
    if (errors > 0)
      printf("segmented reduction does not agree with the reference! %d errors!\n", errors);

    free(d_out);
  }

  #pragma omp target exit data map(delete: d_in[0:num_elements])
  free(d_in);
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }

  const int multiplier = atoi(argv[1]);
  const int repeat     = atoi(argv[2]);
  size_t num_elements  = 16384ULL * multiplier;

  segreduce(num_elements, repeat);
  return 0;
}

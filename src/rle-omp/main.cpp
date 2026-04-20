// OpenMP target offloading port of rle benchmark.
// Run-length encoding using flag marking + exclusive scan + scatter.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

static int parseArg(int argc, char** argv, const char* name, int def)
{
  int len = (int)strlen(name);
  for (int a = 1; a < argc; a++) {
    if (strncmp(argv[a], name, len) == 0 && argv[a][len] == '=')
      return atoi(argv[a] + len + 1);
  }
  return def;
}

static int rle_cpu(const int* in, int n,
                   int* uniq, int* offsets, int* lengths)
{
  if (n == 0) return 0;
  int runs = 0, i = 0;
  while (i < n) {
    int j = i + 1;
    while (j < n && in[j] == in[i]) j++;
    uniq[runs]    = in[i];
    offsets[runs] = i;
    lengths[runs] = j - i;
    runs++;
    i = j;
  }
  return runs;
}

static void initInput(int* h_in, int num_items, int max_segment)
{
  unsigned int max_int = (unsigned int)-1;
  int key = 0, i = 0;
  srand(12345);
  while (i < num_items) {
    int repeat;
    if (max_segment < 2) {
      repeat = 1;
    } else {
      unsigned int r = (unsigned int)rand();
      repeat = (int)((double)r * (double)max_segment / (double)max_int);
      if (repeat < 1) repeat = 1;
    }
    int end = i + repeat;
    if (end > num_items) end = num_items;
    for (int k = i; k < end; k++) h_in[k] = key;
    i = end;
    key++;
  }
}

int main(int argc, char** argv)
{
  int num_items    = parseArg(argc, argv, "--n", 1000000);
  int timing_iters = parseArg(argc, argv, "--i", 1);

  if (num_items    <= 0) num_items    = 1000000;
  if (timing_iters <= 0) timing_iters = 1;

  int* h_in      = new int[num_items];
  int* h_uniq    = new int[num_items];
  int* h_lengths = new int[num_items];
  int* ref_uniq    = new int[num_items];
  int* ref_offsets = new int[num_items];
  int* ref_lengths = new int[num_items];

  // Allocate device arrays
  int* d_in      = new int[num_items];
  int* d_flags   = new int[num_items];
  int* d_scan    = new int[num_items + 1];
  int* d_uniq    = new int[num_items];
  int* d_lengths = new int[num_items];
  int* d_num_runs= new int[1];

  const int max_seg_limit = (num_items < (1 << 16)) ? num_items : (1 << 16);
  for (int max_segment = 1; max_segment <= max_seg_limit; max_segment <<= 4) {
    if (max_segment < 1) max_segment = 1;
    initInput(h_in, num_items, max_segment);

    int ref_runs = rle_cpu(h_in, num_items, ref_uniq, ref_offsets, ref_lengths);
    float avg_run = (ref_runs > 0) ? (float)num_items / ref_runs : (float)num_items;

    printf("\nTest pointer: %d items, %d segments (avg run length %.3f), "
           "{int key, int offset, int length}, max_segment %d\n",
           num_items, ref_runs, avg_run, max_segment);
    fflush(stdout);

    for (int i = 0; i < num_items; i++) d_in[i] = h_in[i];

#pragma omp target enter data map(to: d_in[0:num_items]) \
  map(alloc: d_flags[0:num_items], d_scan[0:num_items+1], \
             d_uniq[0:num_items], d_lengths[0:num_items], d_num_runs[0:1])

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < timing_iters; iter++) {
      // Step 1: mark run starts
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < num_items; i++)
        d_flags[i] = (i == 0 || d_in[i] != d_in[i - 1]) ? 1 : 0;

      // Step 2: exclusive prefix scan using OpenMP scan directive
      {
        int prefix = 0;
#pragma omp target teams distribute parallel for reduction(inscan, +:prefix) thread_limit(256)
        for (int i = 0; i < num_items; i++) {
          d_scan[i] = prefix;
#pragma omp scan exclusive(prefix)
          prefix += d_flags[i];
        }
        // Store total count
        int n = num_items;
#pragma omp target teams distribute parallel for thread_limit(1)
        for (int j = 0; j < 1; j++) {
          // compute last element of scan + last flag
          d_scan[n] = d_scan[n-1] + d_flags[n-1];
          d_num_runs[0] = d_scan[n];
        }
      }

      // Step 3: scatter unique values
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < num_items; i++) {
        if (d_flags[i]) {
          int pos = d_scan[i];
          d_uniq[pos] = d_in[i];
        }
      }

      // Step 4: compute run lengths from flag positions
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < num_items; i++) {
        if (d_flags[i]) {
          int pos = d_scan[i];
          // Find next flag position
          int next_start = num_items;
          for (int j = i + 1; j <= num_items; j++) {
            if (j == num_items || d_flags[j]) { next_start = j; break; }
          }
          d_lengths[pos] = next_start - i;
        }
      }
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

#pragma omp target update from(d_uniq[0:num_items], d_lengths[0:num_items], d_num_runs[0:1])
    for (int i = 0; i < num_items; i++) {
      h_uniq[i]    = d_uniq[i];
      h_lengths[i] = d_lengths[i];
    }
    int got_runs = d_num_runs[0];

    bool ok_keys = (got_runs == ref_runs);
    if (ok_keys)
      for (int i = 0; i < ref_runs; i++)
        if (h_uniq[i] != ref_uniq[i]) { ok_keys = false; break; }

    bool ok_lengths = (got_runs == ref_runs);
    if (ok_lengths)
      for (int i = 0; i < ref_runs; i++)
        if (h_lengths[i] != ref_lengths[i]) { ok_lengths = false; break; }

    printf("\t Keys %s\n",    ok_keys    ? "PASS" : "FAIL");
    printf("\t Lengths %s\n", ok_lengths ? "PASS" : "FAIL");
    printf("\t Count %s\n",   (got_runs == ref_runs) ? "PASS" : "FAIL");

    if (timing_iters > 0) {
      float avg_ms   = (float)time * 1e-6f / timing_iters;
      float giga_rate = (float)num_items / avg_ms / 1e9f;
      int bytes_moved = num_items * (int)sizeof(int)
                      + ref_runs * (int)(sizeof(int) + sizeof(int));
      float giga_bw   = (float)bytes_moved / avg_ms / 1e9f;
      printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s\n\n",
             avg_ms, giga_rate, giga_bw);
    } else {
      printf("\n\n");
    }
    fflush(stdout);

#pragma omp target exit data map(delete: d_in[0:num_items], d_flags[0:num_items], \
  d_scan[0:num_items+1], d_uniq[0:num_items], d_lengths[0:num_items], d_num_runs[0:1])
  }

  delete[] h_in; delete[] h_uniq; delete[] h_lengths;
  delete[] ref_uniq; delete[] ref_offsets; delete[] ref_lengths;
  delete[] d_in; delete[] d_flags; delete[] d_scan;
  delete[] d_uniq; delete[] d_lengths; delete[] d_num_runs;
  return 0;
}

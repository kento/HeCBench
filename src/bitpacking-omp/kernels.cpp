#include "utils.h"
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <climits>

// ---------------------------------------------------------------------------
// Phase 1+2 combined: compute global min and max over the input, then derive
// the required bit-width. Results written to host-side scalar arrays that the
// caller will pass to Phase 3 via map(to:...).
// ---------------------------------------------------------------------------
static void bitPackConfigLaunch(
    int32_t* in, size_t n,
    int32_t* minValOut, unsigned char* numBitsOut)
{
  int32_t gMin = INT32_MAX;
  int32_t gMax = INT32_MIN;

  #pragma omp target teams distribute parallel for \
      reduction(min:gMin) reduction(max:gMax) thread_limit(BLOCK_SIZE)
  for (int i = 0; i < (int)n; i++) {
    int32_t val = in[i];
    if (val < gMin) gMin = val;
    if (val > gMax) gMax = val;
  }

  *minValOut  = gMin;
  uint32_t range = static_cast<uint32_t>(gMax) - static_cast<uint32_t>(gMin);
  *numBitsOut = static_cast<unsigned char>(32 - countLeadingZeros32(range));
}

// ---------------------------------------------------------------------------
// Phase 3: pack each input element into numBits bits in the output array.
// Each output uint32 word is written by one "thread" (global id).
// ---------------------------------------------------------------------------
static void bitPackLaunch(
    unsigned char numBits,
    int32_t       valueOffset,
    uint32_t*     outPtr,
    int32_t*      in,
    size_t        n)
{
  if (numBits == 0) return;  // All identical values — output is already zero-filled.

  const int numBlocks  = (int)roundUpDiv(n, (size_t)BLOCK_SIZE);
  const int gridBlocks = std::min(4096, numBlocks);
  const int nbits      = (int)numBits;
  const int voff       = (int)valueOffset;

  #pragma omp target teams distribute parallel for \
      num_teams(gridBlocks) thread_limit(BLOCK_SIZE)
  for (int global_id = 0; global_id < gridBlocks * BLOCK_SIZE; global_id++) {
    const int blockId    = global_id / BLOCK_SIZE;
    const int threadIdx  = global_id % BLOCK_SIZE;

    for (int blockI = blockId; blockI < numBlocks; blockI += gridBlocks) {
      const int    outputIdx = threadIdx + blockI * BLOCK_SIZE;
      const size_t bitStart  = (size_t)outputIdx * 32U;
      const size_t bitEnd    = bitStart + 32U;

      const int startIdx = (int)std::min(bitStart / (size_t)nbits, n);
      const int endIdx   = (int)std::min(
          roundUpDiv(bitEnd, (size_t)nbits), n);

      uint32_t val = 0;
      for (int idx = startIdx; idx < endIdx; ++idx) {
        uint32_t bits = static_cast<uint32_t>(in[idx] - voff);
        const int offset = (int)(
            static_cast<int64_t>(idx) * nbits
            - static_cast<int64_t>(bitStart));
        if (offset > 0)
          bits <<= offset;
        else
          bits >>= (-offset);
        val |= bits;
      }

      if (startIdx < (int)n)
        outPtr[outputIdx] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// compress: runs the full pipeline 1000 times and reports timing.
// ---------------------------------------------------------------------------
void compress(
    uint32_t*      output,
    int32_t*       input,
    size_t         n,
    int32_t*       minValueDevice,
    unsigned char* numBitsDevice)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < 1000; ++i) {
    bitPackConfigLaunch(input, n, minValueDevice, numBitsDevice);

    const unsigned char numBits   = *numBitsDevice;
    const int32_t       minValue  = *minValueDevice;
    bitPackLaunch(numBits, minValue, output, input, n);
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
  printf("Total kernel execution time (1000 iterations) = %f (s)\n",
         time * 1e-9f);
}

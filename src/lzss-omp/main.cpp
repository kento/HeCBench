// OpenMP target port of lzss-kokkos: LZSS compression/decompression.

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include "utils.h"

#define BLOCK_SIZE   2048
#define WINDOW_SIZE  32

typedef uint32_t INPUT_TYPE;

constexpr int BLOCK_ELEMS = BLOCK_SIZE / (int)sizeof(INPUT_TYPE);

int main(int argc, char *argv[])
{
    std::string inputFileName;
    int opt, repeat = 1;

    while ((opt = getopt(argc, argv, "i:n:h")) != -1) {
        switch (opt) {
        case 'i': inputFileName = optarg;  break;
        case 'n': repeat = atoi(optarg);   break;
        case 'h':
            printf("Usage: ./main -i <inputfile> -n <repeat>\n");
            return 0;
        }
    }

    if (inputFileName.empty()) {
        printf("Usage: ./main -i <inputfile> -n <repeat>\n");
        return 1;
    }

    INPUT_TYPE *hostArray = io::read_binary_to_new_array<INPUT_TYPE>(inputFileName);
    uint32_t fileSize   = (uint32_t)io::FileSize(inputFileName);

    uint32_t paddingSize  = fileSize % BLOCK_SIZE == 0 ? 0
                                                       : BLOCK_SIZE - fileSize % BLOCK_SIZE;
    uint32_t datatypeSize = (fileSize + paddingSize) / sizeof(INPUT_TYPE);
    uint32_t numOfBlocks  = datatypeSize * sizeof(INPUT_TYPE) / BLOCK_SIZE;

    int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

    INPUT_TYPE* deviceArray   = (INPUT_TYPE*)malloc(datatypeSize * sizeof(INPUT_TYPE));
    INPUT_TYPE* deviceOutput  = (INPUT_TYPE*)malloc(datatypeSize * sizeof(INPUT_TYPE));
    uint32_t*   flagArrSize   = (uint32_t*)malloc((numOfBlocks + 1) * sizeof(uint32_t));
    uint32_t*   flagArrOff    = (uint32_t*)malloc((numOfBlocks + 1) * sizeof(uint32_t));
    uint32_t*   compDataSize  = (uint32_t*)malloc((numOfBlocks + 1) * sizeof(uint32_t));
    uint32_t*   compDataOff   = (uint32_t*)malloc((numOfBlocks + 1) * sizeof(uint32_t));
    uint8_t*    tmpFlagArr    = (uint8_t*) malloc(datatypeSize / 8);
    uint8_t*    tmpCompData   = (uint8_t*) malloc((size_t)datatypeSize * sizeof(INPUT_TYPE));
    uint8_t*    flagArr       = (uint8_t*) malloc(datatypeSize / 8);
    uint8_t*    compData      = (uint8_t*) malloc((size_t)datatypeSize * sizeof(INPUT_TYPE));

    memset(deviceArray, 0, datatypeSize * sizeof(INPUT_TYPE));
    memcpy(deviceArray, hostArray, fileSize);

    size_t nb1 = numOfBlocks + 1;
    size_t faSz = datatypeSize / 8;
    size_t cdSz = (size_t)datatypeSize * sizeof(INPUT_TYPE);

    #pragma omp target enter data \
        map(to: deviceArray[0:datatypeSize]) \
        map(alloc: deviceOutput[0:datatypeSize], \
                   flagArrSize[0:nb1], flagArrOff[0:nb1], \
                   compDataSize[0:nb1], compDataOff[0:nb1], \
                   tmpFlagArr[0:faSz], tmpCompData[0:cdSz], \
                   flagArr[0:faSz], compData[0:cdSz])

    auto compStart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
        // Zero device arrays
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (uint32_t i = 0; i < nb1; i++) {
            flagArrSize[i]  = 0;
            compDataSize[i] = 0;
        }
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (size_t i = 0; i < faSz; i++) tmpFlagArr[i] = 0;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (size_t i = 0; i < cdSz; i++) tmpCompData[i] = 0;

        // Kernel I: match-finding + encode (one thread per block)
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int blk = 0; blk < (int)numOfBlocks; blk++) {
            uint8_t lenBuf[BLOCK_ELEMS];
            uint8_t offBuf[BLOCK_ELEMS];

            for (int pos = 0; pos < BLOCK_ELEMS; pos++) {
                int winStart = (pos - WINDOW_SIZE) < 0 ? 0 : pos - WINDOW_SIZE;
                int winPtr = winStart, bufPtr = pos;
                uint8_t maxLen = 0, maxOff = 0, len = 0, off = 0;

                while (winPtr < pos && bufPtr < BLOCK_ELEMS) {
                    if (deviceArray[blk * BLOCK_ELEMS + bufPtr] ==
                        deviceArray[blk * BLOCK_ELEMS + winPtr]) {
                        if (off == 0) off = (uint8_t)(bufPtr - winPtr);
                        len++; bufPtr++;
                    } else {
                        if (len > maxLen) { maxLen = len; maxOff = off; }
                        len = 0; off = 0; bufPtr = pos;
                    }
                    winPtr++;
                }
                if (len > maxLen) { maxLen = len; maxOff = off; }
                lenBuf[pos] = maxLen;
                offBuf[pos] = maxOff;
            }

            uint32_t outOff  = 0;
            uint8_t  flagPos = 0x01, byteFlag = 0;
            uint32_t fc      = 0;
            size_t   tmpBase = (size_t)blk * BLOCK_ELEMS * sizeof(INPUT_TYPE);
            size_t   fBase   = (size_t)blk * (BLOCK_ELEMS / 8);

            int ei = 0;
            while (ei < BLOCK_ELEMS) {
                if (lenBuf[ei] < (uint8_t)minEncodeLength) {
                    const uint8_t *src =
                        (const uint8_t *)&deviceArray[blk * BLOCK_ELEMS + ei];
                    for (int b = 0; b < (int)sizeof(INPUT_TYPE); b++)
                        tmpCompData[tmpBase + outOff + b] = src[b];
                    outOff += (uint32_t)sizeof(INPUT_TYPE);
                    ei++;
                } else {
                    tmpCompData[tmpBase + outOff]     = lenBuf[ei];
                    tmpCompData[tmpBase + outOff + 1] = offBuf[ei];
                    outOff += 2;
                    ei     += lenBuf[ei];
                    byteFlag |= flagPos;
                }
                if (flagPos == 0x80) {
                    tmpFlagArr[fBase + fc++] = byteFlag;
                    flagPos = 0x01; byteFlag = 0;
                    continue;
                }
                flagPos <<= 1;
            }
            if (flagPos != 0x01)
                tmpFlagArr[fBase + fc++] = byteFlag;

            compDataSize[blk] = outOff;
            flagArrSize [blk] = fc;
        }

        // Exclusive prefix-sum of flagArrSize → flagArrOff (host side)
        #pragma omp target update from(flagArrSize[0:nb1], compDataSize[0:nb1])
        flagArrOff[0] = 0;
        for (uint32_t i = 0; i < numOfBlocks; i++)
            flagArrOff[i + 1] = flagArrOff[i] + flagArrSize[i];
        compDataOff[0] = 0;
        for (uint32_t i = 0; i < numOfBlocks; i++)
            compDataOff[i + 1] = compDataOff[i] + compDataSize[i];
        #pragma omp target update to(flagArrOff[0:nb1], compDataOff[0:nb1])

        // Kernel III: pack into contiguous output arrays
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int blk = 0; blk < (int)numOfBlocks; blk++) {
            int fOff = (int)flagArrOff [blk];
            int fSz  = (int)flagArrOff [blk + 1] - fOff;
            int dOff = (int)compDataOff[blk];
            int dSz  = (int)compDataOff[blk + 1] - dOff;

            size_t fBase   = (size_t)blk * (BLOCK_ELEMS / 8);
            size_t tmpBase = (size_t)blk * BLOCK_ELEMS * sizeof(INPUT_TYPE);

            for (int i = 0; i < fSz; i++)
                flagArr [fOff + i] = tmpFlagArr [fBase   + i];
            for (int i = 0; i < dSz; i++)
                compData[dOff + i] = tmpCompData[tmpBase + i];
        }
    }

    auto compStop = std::chrono::steady_clock::now();

    // Decompression
    auto decompStart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (uint32_t i = 0; i < datatypeSize; i++) deviceOutput[i] = 0;

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int tid = 0; tid < (int)numOfBlocks; tid++) {
            int      fOff = (int)flagArrOff [tid];
            int      fSz  = (int)flagArrOff [tid + 1] - fOff;
            int      dOff = (int)compDataOff[tid];
            uint32_t di   = 0, ci = 0;

            for (int fi = 0; fi < fSz; fi++) {
                uint8_t byteFlag = flagArr[fOff + fi];
                for (int bit = 0; bit < 8; bit++) {
                    if ((byteFlag >> bit) & 1) {
                        int len = (int)compData[dOff + ci];
                        int off = (int)compData[dOff + ci + 1];
                        ci += 2;
                        uint32_t dstart = di;
                        for (int k = 0; k < len; k++) {
                            deviceOutput[tid * BLOCK_ELEMS + di] =
                                deviceOutput[tid * BLOCK_ELEMS + dstart - off + k];
                            di++;
                        }
                    } else {
                        uint8_t *dst = (uint8_t *)&deviceOutput[tid * BLOCK_ELEMS + di];
                        for (int b = 0; b < (int)sizeof(INPUT_TYPE); b++)
                            dst[b] = compData[dOff + ci + b];
                        ci += sizeof(INPUT_TYPE);
                        di++;
                    }
                    if (di >= (uint32_t)BLOCK_ELEMS) goto next_block;
                }
            }
            next_block:;
        }
    }

    auto decompStop = std::chrono::steady_clock::now();

    #pragma omp target update from(deviceOutput[0:datatypeSize])

    bool ok = true;
    for (uint32_t i = 0; i < fileSize / sizeof(INPUT_TYPE); i++) {
        if (hostArray[i] != deviceOutput[i]) {
            printf("verification failed at index %u\n", i);
            ok = false;
            break;
        }
    }
    if (ok) printf("verification passed\n");

    float origSz = (float)fileSize;
    float compSz = (float)(sizeof(uint32_t) * (numOfBlocks + 1) * 2
                           + flagArrOff[numOfBlocks] + compDataOff[numOfBlocks]);
    std::cout << "compression ratio: " << (origSz / compSz) << "\n";

    float compMs  = std::chrono::duration<float, std::milli>(compStop  - compStart ).count();
    float decompMs = std::chrono::duration<float, std::milli>(decompStop - decompStart).count();
    std::cout << "compression e2e throughput: "
              << origSz / 1024.f / 1024.f / (compMs  / repeat) << " GB/s\n";
    std::cout << "decompression e2e throughput: "
              << origSz / 1024.f / 1024.f / (decompMs / repeat) << " GB/s\n";

    #pragma omp target exit data \
        map(delete: deviceArray[0:datatypeSize], deviceOutput[0:datatypeSize], \
                    flagArrSize[0:nb1], flagArrOff[0:nb1], \
                    compDataSize[0:nb1], compDataOff[0:nb1], \
                    tmpFlagArr[0:faSz], tmpCompData[0:cdSz], \
                    flagArr[0:faSz], compData[0:cdSz])

    free(deviceArray); free(deviceOutput);
    free(flagArrSize); free(flagArrOff);
    free(compDataSize); free(compDataOff);
    free(tmpFlagArr); free(tmpCompData);
    free(flagArr); free(compData);
    delete[] hostArray;
    return 0;
}

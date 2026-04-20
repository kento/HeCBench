// Smith-Waterman sequence alignment benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <string>

static const short BLOSUM62[24 * 24] = {
    4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4,
   -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4,
   -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4,
   -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4,
    0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
   -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4,
   -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
    0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4,
   -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4,
   -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4,
   -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4,
   -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4,
   -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4,
   -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4,
   -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
    1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4,
    0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4,
   -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4,
   -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4,
    0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4,
   -2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4,
   -1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
    0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4,
   -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1
};

static const short AA_ENCODE[128] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    23,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,20,4,3,6,13,7,8,9,0,11,10,12,2,0,14,5,1,15,16,0,19,17,22,18,21,
    0
};

static const char AA_CHARS[] = "ACDEFGHIKLMNPQRSTVWY";
static constexpr int NUM_AA   = 20;
static constexpr short GAP_OPEN   = -6;
static constexpr short GAP_EXTEND = -1;
static constexpr int MAX_SEQ_LEN = 512;

#pragma omp declare target
short sw_align(const char* seqA, int lenA,
               const char* seqB, int lenB,
               const short* blosum,
               const short* enc,
               short* H_prev,
               short* H_curr,
               short* F_arr,
               short& best_i, short& best_j)
{
    short best = 0;
    best_i = 0; best_j = 0;

    for (int j = 0; j <= lenB; ++j) {
        H_prev[j] = 0;
        F_arr[j]  = 0;
    }

    for (int i = 1; i <= lenA; ++i) {
        H_curr[0] = 0;
        short E   = 0;
        const short enc_ai = enc[(int)(unsigned char)seqA[i-1]];

        for (int j = 1; j <= lenB; ++j) {
            const short F = (H_prev[j] + GAP_OPEN > F_arr[j] + GAP_EXTEND)
                            ? H_prev[j] + GAP_OPEN
                            : F_arr[j]  + GAP_EXTEND;
            F_arr[j] = F;
            const short E_new = (H_curr[j-1] + GAP_OPEN > E + GAP_EXTEND)
                                ? H_curr[j-1] + GAP_OPEN
                                : E           + GAP_EXTEND;
            E = E_new;
            const short enc_bj  = enc[(int)(unsigned char)seqB[j-1]];
            const short sub     = blosum[enc_ai * 24 + enc_bj];
            const short diag    = H_prev[j-1] + sub;
            short h = 0;
            if (diag > h) h = diag;
            if (F    > h) h = F;
            if (E    > h) h = E;
            H_curr[j] = h;
            if (h > best) {
                best   = h;
                best_i = (short)i;
                best_j = (short)j;
            }
        }
        for (int j = 0; j <= lenB; ++j) H_prev[j] = H_curr[j];
    }
    return best;
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
    int numPairs  = 100;
    int avgLength = 100;
    int repeat    = 100;

    if (argc >= 2) numPairs  = std::atoi(argv[1]);
    if (argc >= 3) avgLength = std::atoi(argv[2]);
    if (argc >= 4) repeat    = std::atoi(argv[3]);

    const int minLen = (avgLength / 2 > 1) ? avgLength / 2 : 1;
    const int maxLen = (3 * avgLength / 2 < MAX_SEQ_LEN) ? 3 * avgLength / 2 : MAX_SEQ_LEN;

    // Flat 1D arrays for 2D data
    char*  seqA_d  = (char*) malloc((size_t)numPairs * maxLen * sizeof(char));
    char*  seqB_d  = (char*) malloc((size_t)numPairs * maxLen * sizeof(char));
    int*   lenA_d  = (int*)  malloc(numPairs * sizeof(int));
    int*   lenB_d  = (int*)  malloc(numPairs * sizeof(int));
    short* scores_d= (short*)malloc(numPairs * sizeof(short));
    short* endA_d  = (short*)malloc(numPairs * sizeof(short));
    short* endB_d  = (short*)malloc(numPairs * sizeof(short));
    short* blosum_d= (short*)malloc(24 * 24 * sizeof(short));
    short* enc_d   = (short*)malloc(128 * sizeof(short));
    short* H_prev_d= (short*)malloc((size_t)numPairs * (MAX_SEQ_LEN + 1) * sizeof(short));
    short* H_curr_d= (short*)malloc((size_t)numPairs * (MAX_SEQ_LEN + 1) * sizeof(short));
    short* F_arr_d = (short*)malloc((size_t)numPairs * (MAX_SEQ_LEN + 1) * sizeof(short));

    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> len_dist(minLen, maxLen);
        std::uniform_int_distribution<int> aa_dist(0, NUM_AA - 1);
        for (int p = 0; p < numPairs; ++p) {
            lenA_d[p] = len_dist(rng);
            lenB_d[p] = len_dist(rng);
            for (int i = 0; i < lenA_d[p]; ++i) seqA_d[p * maxLen + i] = AA_CHARS[aa_dist(rng)];
            for (int j = 0; j < lenB_d[p]; ++j) seqB_d[p * maxLen + j] = AA_CHARS[aa_dist(rng)];
        }
        for (int i = 0; i < 24*24; ++i) blosum_d[i] = BLOSUM62[i];
        for (int i = 0; i < 128;   ++i) enc_d[i]    = AA_ENCODE[i];
    }

    const size_t seqA_sz   = (size_t)numPairs * maxLen;
    const size_t seqB_sz   = (size_t)numPairs * maxLen;
    const size_t scratch_sz= (size_t)numPairs * (MAX_SEQ_LEN + 1);

    #pragma omp target enter data map(alloc: seqA_d[0:seqA_sz], seqB_d[0:seqB_sz], \
        lenA_d[0:numPairs], lenB_d[0:numPairs], scores_d[0:numPairs], \
        endA_d[0:numPairs], endB_d[0:numPairs], blosum_d[0:576], enc_d[0:128], \
        H_prev_d[0:scratch_sz], H_curr_d[0:scratch_sz], F_arr_d[0:scratch_sz])
    #pragma omp target update to(seqA_d[0:seqA_sz], seqB_d[0:seqB_sz], \
        lenA_d[0:numPairs], lenB_d[0:numPairs], blosum_d[0:576], enc_d[0:128])

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < repeat; ++rep) {
        const int ml = maxLen;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int p = 0; p < numPairs; p++) {
            const int la = lenA_d[p];
            const int lb = lenB_d[p];
            short* Hp = &H_prev_d[p * (MAX_SEQ_LEN + 1)];
            short* Hc = &H_curr_d[p * (MAX_SEQ_LEN + 1)];
            short* Fp = &F_arr_d [p * (MAX_SEQ_LEN + 1)];
            short ei = 0, ej = 0;
            scores_d[p] = sw_align(
                &seqA_d[p * ml], la,
                &seqB_d[p * ml], lb,
                blosum_d, enc_d,
                Hp, Hc, Fp, ei, ej);
            endA_d[p] = ei;
            endB_d[p] = ej;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    #pragma omp target update from(scores_d[0:numPairs], endA_d[0:numPairs], endB_d[0:numPairs])

    const int show = (numPairs < 5) ? numPairs : 5;
    printf("BSW OpenMP benchmark\n");
    printf("Pairs: %d, avgLen: %d, repeat: %d\n", numPairs, avgLength, repeat);
    printf("Sample results (pair, score, endA, endB):\n");
    for (int p = 0; p < show; ++p)
        printf("  pair %3d  score=%5d  endA=%4d  endB=%4d\n",
               p, scores_d[p], endA_d[p], endB_d[p]);
    printf("Total time       : %.3f s\n", elapsed);
    printf("Avg per repeat   : %.3f ms\n", elapsed * 1e3 / repeat);
    printf("Throughput       : %.1f GCUPS\n",
           (double)numPairs * avgLength * avgLength * repeat / elapsed / 1e9);

    #pragma omp target exit data map(delete: seqA_d[0:seqA_sz], seqB_d[0:seqB_sz], \
        lenA_d[0:numPairs], lenB_d[0:numPairs], scores_d[0:numPairs], \
        endA_d[0:numPairs], endB_d[0:numPairs], blosum_d[0:576], enc_d[0:128], \
        H_prev_d[0:scratch_sz], H_curr_d[0:scratch_sz], F_arr_d[0:scratch_sz])

    free(seqA_d); free(seqB_d); free(lenA_d); free(lenB_d);
    free(scores_d); free(endA_d); free(endB_d);
    free(blosum_d); free(enc_d);
    free(H_prev_d); free(H_curr_d); free(F_arr_d);
    return 0;
}

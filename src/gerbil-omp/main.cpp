// OpenMP target offloading port of gerbil-kokkos (k-mer counting with hash table)

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>

static const int HT_SIZE = 1 << 24; // 16M

#pragma omp declare target
static inline int base_encode(char c)
{
    if (c == 'A' || c == 'a') return 0;
    if (c == 'C' || c == 'c') return 1;
    if (c == 'G' || c == 'g') return 2;
    return 3;
}

static inline unsigned int kmer_hash(uint64_t kmer)
{
    kmer = (~kmer) + (kmer << 21);
    kmer =   kmer  ^ (kmer >> 24);
    kmer = ( kmer  + (kmer << 3)) + (kmer << 8);
    kmer =   kmer  ^ (kmer >> 14);
    kmer = ( kmer  + (kmer << 2)) + (kmer << 4);
    kmer =   kmer  ^ (kmer >> 28);
    kmer =   kmer  + (kmer << 31);
    return (unsigned int)kmer;
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
    int numSeq  = 1000;
    int seqLen  = 1000;
    int k       = 32;
    int repeat  = 1;

    if (argc > 1) numSeq  = atoi(argv[1]);
    if (argc > 2) seqLen  = atoi(argv[2]);
    if (argc > 3) k       = atoi(argv[3]);
    if (argc > 4) repeat  = atoi(argv[4]);

    if (k > 32) { fprintf(stderr, "k must be <= 32\n"); return 1; }
    if (seqLen < k) { fprintf(stderr, "seqLen must be >= k\n"); return 1; }

    printf("K-mer length: %d\n", k);

    const long long kmers_per_seq = seqLen - k + 1;
    const long long total_kmers   = (long long)numSeq * kmers_per_seq;
    printf("Total k-mers: %lld\n", total_kmers);

    const int totalChars = numSeq * seqLen;
    char* d_seqs = (char*)malloc(totalChars * sizeof(char));
    int*  d_ht   = (int*) malloc(HT_SIZE   * sizeof(int));

    for (int i = 0; i < HT_SIZE; i++) d_ht[i] = 0;

    #pragma omp target enter data map(alloc: d_seqs[0:totalChars]) map(tofrom: d_ht[0:HT_SIZE])

    // Generate random sequences on device
    const int sl = seqLen;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int s = 0; s < numSeq; s++) {
        uint64_t state = (uint64_t)(s + 1) * 6364136223846793005ULL;
        const char bases[4] = {'A', 'C', 'G', 'T'};
        for (int p = 0; p < sl; p++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            d_seqs[s * sl + p] = bases[(state >> 33) & 3];
        }
    }

    double total_ms = 0.0;

    for (int rep = 0; rep < repeat; rep++) {
        // Zero hash table
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < HT_SIZE; i++) d_ht[i] = 0;

        auto t1 = std::chrono::steady_clock::now();

        const int kk = k;
        const int ht_mask = HT_SIZE - 1;

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int s = 0; s < numSeq; s++) {
            const int base_off = s * sl;
            uint64_t kmer    = 0;
            const uint64_t mask = (kk < 32) ? ((1ULL << (2*kk)) - 1ULL) : ~0ULL;
            for (int p = 0; p < kk; p++) {
                kmer = ((kmer << 2) | (uint64_t)base_encode(d_seqs[base_off + p])) & mask;
            }
            unsigned int idx = kmer_hash(kmer) & ht_mask;
            #pragma omp atomic update
            d_ht[idx] += 1;

            for (int p = kk; p < sl; p++) {
                kmer = ((kmer << 2) | (uint64_t)base_encode(d_seqs[base_off + p])) & mask;
                idx  = kmer_hash(kmer) & ht_mask;
                #pragma omp atomic update
                d_ht[idx] += 1;
            }
        }

        auto t2 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    long long unique_kmers = 0;
    #pragma omp target teams distribute parallel for reduction(+:unique_kmers) thread_limit(256)
    for (int i = 0; i < HT_SIZE; i++) {
        if (d_ht[i] > 0) unique_kmers++;
    }

    printf("Unique k-mers: %lld\n", unique_kmers);
    printf("Kernel time: %.3f (ms)\n", total_ms / repeat);

    #pragma omp target exit data map(delete: d_seqs[0:totalChars], d_ht[0:HT_SIZE])
    free(d_seqs); free(d_ht);
    return 0;
}

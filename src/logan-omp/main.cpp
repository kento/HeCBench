// OpenMP target offloading port of logan-kokkos benchmark
// LOGAN: x-drop seed-and-extend alignment algorithm
// Original: A. Zeni, G. Guidi
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdlib>

#define NOW std::chrono::high_resolution_clock::now()

// ---------------------------------------------------------------------------
// Scoring scheme
// ---------------------------------------------------------------------------
struct ScoringSchemeL {
  int match_score;
  int mismatch_score;
  int gap_extend_score;
  int gap_open_score;

  ScoringSchemeL()
    : match_score(1), mismatch_score(-1), gap_extend_score(-1), gap_open_score(-1) {}

  ScoringSchemeL(int match, int mismatch, int gap_extend, int gap_open)
    : match_score(match), mismatch_score(mismatch),
      gap_extend_score(gap_extend), gap_open_score(gap_open) {}
};

inline int scoreGapExtend(const ScoringSchemeL& me) { return me.gap_extend_score; }
inline int scoreGapOpen  (const ScoringSchemeL& me) { return me.gap_open_score;   }

// ---------------------------------------------------------------------------
// Extension direction
// ---------------------------------------------------------------------------
enum ExtensionDirectionL {
  EXTEND_NONEL  = 0,
  EXTEND_LEFTL  = 1,
  EXTEND_RIGHTL = 2,
  EXTEND_BOTHL  = 3
};

// DP scoring constants (linear gap)
static constexpr int MATCH    =  1;
static constexpr int MISMATCH = -1;
static constexpr int GAP_EXT  = -1;
static constexpr int GAP_OPEN = -1;
static constexpr short UNDEF  = -32767;
static constexpr short MIN_VAL= -32768;

// ---------------------------------------------------------------------------
// Seed
// ---------------------------------------------------------------------------
struct SeedL {
  int beginPositionH, beginPositionV;
  int endPositionH,   endPositionV;
  int seedLength;
  int lowerDiagonal, upperDiagonal;
  int beginDiagonal,  endDiagonal;
  int score;
};

// ---------------------------------------------------------------------------
// All device-callable functions
// ---------------------------------------------------------------------------
#pragma omp declare target

inline void SeedL_init(SeedL& s) {
  s.beginPositionH = 0; s.beginPositionV = 0;
  s.endPositionH   = 0; s.endPositionV   = 0;
  s.lowerDiagonal  = 0; s.upperDiagonal  = 0;
  s.score = 0;
}

inline void SeedL_init(SeedL& s, int bH, int bV, int sLen) {
  s.beginPositionH = bH;         s.beginPositionV = bV;
  s.endPositionH   = bH + sLen;  s.endPositionV   = bV + sLen;
  s.lowerDiagonal  = bH - bV;    s.upperDiagonal  = bH - bV;
  s.beginDiagonal  = bH - bV;    s.endDiagonal    = (bH+sLen) - (bV+sLen);
  s.score = 0;
}

inline int scoreGap(const ScoringSchemeL& me) { return me.gap_open_score; }
inline int scoreMatch(const ScoringSchemeL& me) { return me.match_score; }
inline int scoreMismatch(const ScoringSchemeL& me) { return me.mismatch_score; }
inline int score(const ScoringSchemeL& me, char h, char v) {
  return (h == v) ? me.match_score : me.mismatch_score;
}

template<class T>
inline const T& max_logan(const T& a, const T& b) { return (a < b) ? b : a; }
template<class T>
inline const T& min_logan(const T& a, const T& b) { return (a < b) ? a : b; }

inline int getBeginPositionH(const SeedL& s) { return s.beginPositionH; }
inline int getBeginPositionV(const SeedL& s) { return s.beginPositionV; }
inline int getEndPositionH  (const SeedL& s) { return s.endPositionH;   }
inline int getEndPositionV  (const SeedL& s) { return s.endPositionV;   }
inline int getLowerDiagonal (const SeedL& s) { return s.lowerDiagonal;  }
inline int getUpperDiagonal (const SeedL& s) { return s.upperDiagonal;  }
inline int getBeginDiagonal (const SeedL& s) { return s.beginDiagonal;  }
inline int getEndDiagonal   (const SeedL& s) { return s.endDiagonal;    }

inline void setBeginPositionH(SeedL& s, int v) { s.beginPositionH = v; }
inline void setBeginPositionV(SeedL& s, int v) { s.beginPositionV = v; }
inline void setEndPositionH  (SeedL& s, int v) { s.endPositionH   = v; }
inline void setEndPositionV  (SeedL& s, int v) { s.endPositionV   = v; }
inline void setLowerDiagonal (SeedL& s, int v) { s.lowerDiagonal  = v; }
inline void setUpperDiagonal (SeedL& s, int v) { s.upperDiagonal  = v; }
inline void setBeginDiagonal (SeedL& s, int v) { s.beginDiagonal  = v; }
inline void setEndDiagonal   (SeedL& s, int v) { s.endDiagonal    = v; }

inline void updateExtendedSeedL(SeedL& seed, ExtensionDirectionL dir,
                                 int cols, int rows,
                                 int lowerDiag, int upperDiag) {
  if (dir == EXTEND_LEFTL) {
    int beginDiag = seed.beginDiagonal;
    if (getLowerDiagonal(seed) > beginDiag + lowerDiag)
      setLowerDiagonal(seed, beginDiag + lowerDiag);
    if (getUpperDiagonal(seed) < beginDiag + upperDiag)
      setUpperDiagonal(seed, beginDiag + upperDiag);
    seed.beginPositionH -= rows;
    seed.beginPositionV -= cols;
  } else {
    int endDiag = seed.endDiagonal;
    if (getUpperDiagonal(seed) < endDiag - lowerDiag)
      setUpperDiagonal(seed, endDiag - lowerDiag);
    if (getLowerDiagonal(seed) > endDiag - upperDiag)
      setLowerDiagonal(seed, endDiag - upperDiag);
    seed.endPositionH += rows;
    seed.endPositionV += cols;
  }
}

inline void calcExtendedLowerDiag(int& lowerDiag, int minCol, int antiDiagNo) {
  int minRow = antiDiagNo - minCol;
  if (minCol - minRow < lowerDiag) lowerDiag = minCol - minRow;
}

inline void calcExtendedUpperDiag(int& upperDiag, int maxCol, int antiDiagNo) {
  int maxRow = antiDiagNo + 1 - maxCol;
  if (maxCol - 1 - maxRow > upperDiag) upperDiag = maxCol - 1 - maxRow;
}

inline void initAntiDiag3(short* ad3, int& a3size,
                           int offset, int maxCol,
                           int antiDiagNo, int minScore,
                           int gapCost, short undef) {
  a3size = maxCol + 1 - offset;
  ad3[0] = undef;
  ad3[maxCol - offset] = undef;
  if (antiDiagNo * gapCost > minScore) {
    if (offset == 0)              ad3[0]             = (short)(antiDiagNo * gapCost);
    if (antiDiagNo - maxCol == 0) ad3[maxCol-offset] = (short)(antiDiagNo * gapCost);
  }
}

inline void computeAntidiag_seq(const short* ad1,
                                 const short* ad2,
                                       short* ad3,
                                 const char* querySeg,
                                 const char* dbSeg,
                                 int best, int scoreDropOff,
                                 int cols, int rows,
                                 int minCol, int maxCol,
                                 int antiDiagNo,
                                 int offset1, int offset2,
                                 ExtensionDirectionL /*dir*/) {
  for (int col = minCol; col < maxCol; col++) {
    int queryPos = col - 1;
    int dbPos    = col + rows - antiDiagNo - 1;

    int tmp = max_logan((int)ad2[col-offset2],
                        (int)ad2[col-offset2-1]) + GAP_EXT;
    int sc  = (querySeg[queryPos] == dbSeg[dbPos]) ? MATCH : MISMATCH;
    tmp     = max_logan((int)ad1[col-offset1-1] + sc, tmp);

    ad3[col - minCol + 1] = (tmp < best - scoreDropOff) ? UNDEF : (short)tmp;
  }
}

// Single-alignment x-drop extension (one direction, sequential).
// Returns the extension score (UNDEF if nothing extended).
// Writes updated seed back.
inline int extendOneDirection(SeedL& seed,
                               const char* querySeg,
                               const char* dbSeg,
                               int qLen, int tLen,
                               ExtensionDirectionL dir,
                               int scoreDropOff,
                               short* antidiag_buf) {
  int cols = qLen + 1;
  int rows = tLen + 1;
  if (rows <= 1 || cols <= 1) return UNDEF;

  const int ADBUF = cols + rows + 4;
  short* ad1 = antidiag_buf + 0*ADBUF;
  short* ad2 = antidiag_buf + 1*ADBUF;
  short* ad3 = antidiag_buf + 2*ADBUF;

  int a1size = 0, a2size = 1, a3size = 2;
  ad2[0] = 0;
  ad3[0] = (short)GAP_EXT;
  ad3[1] = (short)GAP_EXT;

  int antiDiagNo = 1;
  int best = 0;
  int lowerDiag = 0, upperDiag = 0;
  int minCol = 1, maxCol = 2;
  int offset1 = 0, offset2 = 0, offset3 = 0;

  while (minCol < maxCol) {
    ++antiDiagNo;

    // Rotate: ad2→ad1, ad3→ad2, ad1→ad3
    short* t = ad1; ad1 = ad2; ad2 = ad3; ad3 = t;
    int tl = a1size; a1size = a2size; a2size = a3size; a3size = tl;
    offset1 = offset2;
    offset2 = offset3;
    offset3 = minCol - 1;

    initAntiDiag3(ad3, a3size, offset3, maxCol, antiDiagNo,
                  best - scoreDropOff, GAP_EXT, UNDEF);

    computeAntidiag_seq(ad1, ad2, ad3, querySeg, dbSeg,
                        best, scoreDropOff, cols, rows,
                        minCol, maxCol, antiDiagNo,
                        offset1, offset2, dir);

    short antiDiagBest = UNDEF;
    for (int i = 0; i < a3size; i++)
      if (ad3[i] > antiDiagBest) antiDiagBest = ad3[i];
    if (antiDiagBest > best) best = antiDiagBest;

    while (minCol - offset3 < a3size &&
           ad3[minCol - offset3] == UNDEF &&
           minCol - offset2 - 1 < a2size &&
           ad2[minCol - offset2 - 1] == UNDEF)
      ++minCol;

    while (maxCol - offset3 > 0 &&
           ad3[maxCol - offset3 - 1] == UNDEF &&
           ad2[maxCol - offset2 - 1] == UNDEF)
      --maxCol;
    ++maxCol;

    calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
    calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

    minCol = max_logan(minCol, antiDiagNo + 2 - rows);
    maxCol = min_logan(maxCol, cols);
  }

  int longestCol   = a3size + offset3 - 2;
  int longestRow   = antiDiagNo - longestCol;
  short longestScore = ad3[longestCol - offset3];

  if (longestScore == UNDEF) {
    if (ad2[a2size-2] != UNDEF) {
      longestCol   = a2size + offset2 - 2;
      longestRow   = antiDiagNo - 1 - longestCol;
      longestScore = ad2[longestCol - offset2];
    } else if (a2size > 2 && ad2[a2size-3] != UNDEF) {
      longestCol   = a2size + offset2 - 3;
      longestRow   = antiDiagNo - 1 - longestCol;
      longestScore = ad2[longestCol - offset2];
    }
  }

  if (longestScore == UNDEF) {
    for (int i = 0; i < a1size; i++) {
      if (ad1[i] > longestScore) {
        longestScore = ad1[i];
        longestCol   = i + offset1;
        longestRow   = antiDiagNo - 2 - longestCol;
      }
    }
  }

  if (longestScore != UNDEF)
    updateExtendedSeedL(seed, dir, longestCol, longestRow, lowerDiag, upperDiag);

  return (int)longestScore;
}

#pragma omp end declare target

// ---------------------------------------------------------------------------
// OpenMP target offloading: process batch of alignments in parallel
// ---------------------------------------------------------------------------
void extendSeedL_omp(std::vector<SeedL>& seeds,
                     std::vector<std::string>& target,
                     std::vector<std::string>& query,
                     int XDrop, int kmer_length,
                     int* res, int numAlignments) {
  // ---- Compute prefix offsets for packed sequence arrays ----
  std::vector<int> offLeftQ(numAlignments), offLeftT(numAlignments);
  std::vector<int> offRightQ(numAlignments), offRightT(numAlignments);

  int antLenLeft = 0, antLenRight = 0;
  for (int j = 0; j < numAlignments; j++) {
    offLeftQ[j]  = getBeginPositionV(seeds[j]);
    offLeftT[j]  = getBeginPositionH(seeds[j]);
    offRightQ[j] = (int)query[j].size()  - getEndPositionV(seeds[j]);
    offRightT[j] = (int)target[j].size() - getEndPositionH(seeds[j]);
    antLenLeft  = std::max(antLenLeft,  std::min(offLeftQ[j],  offLeftT[j]));
    antLenRight = std::max(antLenRight, std::min(offRightQ[j], offRightT[j]));
  }

  // Prefix sums (cumulative lengths)
  std::vector<int> prefixLQ(numAlignments), prefixLT(numAlignments);
  std::vector<int> prefixRQ(numAlignments), prefixRT(numAlignments);
  std::partial_sum(offLeftQ.begin(),  offLeftQ.end(),  prefixLQ.begin());
  std::partial_sum(offLeftT.begin(),  offLeftT.end(),  prefixLT.begin());
  std::partial_sum(offRightQ.begin(), offRightQ.end(), prefixRQ.begin());
  std::partial_sum(offRightT.begin(), offRightT.end(), prefixRT.begin());

  int totalLQ = prefixLQ[numAlignments-1];
  int totalLT = prefixLT[numAlignments-1];
  int totalRQ = prefixRQ[numAlignments-1];
  int totalRT = prefixRT[numAlignments-1];

  // Pack sequence prefixes/suffixes into flat arrays
  int n_prefQ = totalLQ > 0 ? totalLQ : 1;
  int n_prefT = totalLT > 0 ? totalLT : 1;
  int n_suffQ = totalRQ > 0 ? totalRQ : 1;
  int n_suffT = totalRT > 0 ? totalRT : 1;

  char* prefQ = (char*)malloc(n_prefQ * sizeof(char));
  char* prefT = (char*)malloc(n_prefT * sizeof(char));
  char* suffQ = (char*)malloc(n_suffQ * sizeof(char));
  char* suffT = (char*)malloc(n_suffT * sizeof(char));

  // alignment 0
  std::reverse_copy(query[0].c_str(),  query[0].c_str()  + offLeftQ[0],  prefQ);
  std::memcpy(prefT, target[0].c_str(), offLeftT[0]);
  std::memcpy(suffQ, query[0].c_str()  + getEndPositionV(seeds[0]), offRightQ[0]);
  std::reverse_copy(target[0].c_str() + getEndPositionH(seeds[0]),
                    target[0].c_str() + getEndPositionH(seeds[0]) + offRightT[0],
                    suffT);

  for (int j = 1; j < numAlignments; j++) {
    std::reverse_copy(query[j].c_str(), query[j].c_str() + (prefixLQ[j]-prefixLQ[j-1]),
                      prefQ + prefixLQ[j-1]);
    std::memcpy(prefT + prefixLT[j-1], target[j].c_str(),
                prefixLT[j]-prefixLT[j-1]);
    std::memcpy(suffQ + prefixRQ[j-1],
                query[j].c_str() + getEndPositionV(seeds[j]),
                prefixRQ[j]-prefixRQ[j-1]);
    std::reverse_copy(
      target[j].c_str() + getEndPositionH(seeds[j]),
      target[j].c_str() + getEndPositionH(seeds[j]) + (prefixRT[j]-prefixRT[j-1]),
      suffT + prefixRT[j-1]);
  }

  // Offset arrays
  int* offLQ = (int*)malloc(numAlignments * sizeof(int));
  int* offLT = (int*)malloc(numAlignments * sizeof(int));
  int* offRQ = (int*)malloc(numAlignments * sizeof(int));
  int* offRT = (int*)malloc(numAlignments * sizeof(int));

  // Seed arrays (one copy for left extension, one for right)
  SeedL* seeds_l = (SeedL*)malloc(numAlignments * sizeof(SeedL));
  SeedL* seeds_r = (SeedL*)malloc(numAlignments * sizeof(SeedL));
  int*   scL     = (int*)malloc(numAlignments * sizeof(int));
  int*   scR     = (int*)malloc(numAlignments * sizeof(int));

  for (int j = 0; j < numAlignments; j++) {
    offLQ[j] = prefixLQ[j];
    offLT[j] = prefixLT[j];
    offRQ[j] = prefixRQ[j];
    offRT[j] = prefixRT[j];
    seeds_l[j] = seeds[j];
    seeds_r[j] = seeds[j];
  }

  // Per-alignment antidiag scratch buffers
  int adbuf_max_left  = 0, adbuf_max_right = 0;
  for (int j = 0; j < numAlignments; j++) {
    int bl = offLeftQ[j]  + offLeftT[j]  + 4;
    int br = offRightQ[j] + offRightT[j] + 4;
    if (bl > adbuf_max_left)  adbuf_max_left  = bl;
    if (br > adbuf_max_right) adbuf_max_right = br;
  }
  if (adbuf_max_left  == 0) adbuf_max_left  = 4;
  if (adbuf_max_right == 0) adbuf_max_right = 4;

  size_t ant_l_size = (size_t)numAlignments * adbuf_max_left  * 3;
  size_t ant_r_size = (size_t)numAlignments * adbuf_max_right * 3;
  short* ant_l = (short*)malloc(ant_l_size * sizeof(short));
  short* ant_r = (short*)malloc(ant_r_size * sizeof(short));

  auto start_c = NOW;

  // ---- Run both kernels inside a structured target data region ----
  #pragma omp target data \
    map(to: prefQ[0:n_prefQ], prefT[0:n_prefT], \
            suffQ[0:n_suffQ], suffT[0:n_suffT], \
            offLQ[0:numAlignments], offLT[0:numAlignments], \
            offRQ[0:numAlignments], offRT[0:numAlignments]) \
    map(tofrom: seeds_l[0:numAlignments], seeds_r[0:numAlignments]) \
    map(from: scL[0:numAlignments], scR[0:numAlignments]) \
    map(alloc: ant_l[0:ant_l_size], ant_r[0:ant_r_size])
  {
    // ---- LEFT extension kernel ----
    {
      int abL = adbuf_max_left;
      #pragma omp target teams distribute parallel for \
        num_teams(numAlignments) thread_limit(1)
      for (int myId = 0; myId < numAlignments; myId++) {
        const char* querySeg = (myId == 0) ? prefQ : prefQ + offLQ[myId-1];
        const char* dbSeg    = (myId == 0) ? prefT : prefT + offLT[myId-1];
        int qLen = (myId == 0) ? offLQ[myId] : offLQ[myId] - offLQ[myId-1];
        int tLen = (myId == 0) ? offLT[myId] : offLT[myId] - offLT[myId-1];

        short* buf = ant_l + (size_t)myId * abL * 3;
        scL[myId] = extendOneDirection(seeds_l[myId], querySeg, dbSeg,
                                       qLen, tLen,
                                       EXTEND_LEFTL, XDrop, buf);
      }
    }

    // ---- RIGHT extension kernel ----
    {
      int abR = adbuf_max_right;
      #pragma omp target teams distribute parallel for \
        num_teams(numAlignments) thread_limit(1)
      for (int myId = 0; myId < numAlignments; myId++) {
        const char* querySeg = (myId == 0) ? suffQ : suffQ + offRQ[myId-1];
        const char* dbSeg    = (myId == 0) ? suffT : suffT + offRT[myId-1];
        int qLen = (myId == 0) ? offRQ[myId] : offRQ[myId] - offRQ[myId-1];
        int tLen = (myId == 0) ? offRT[myId] : offRT[myId] - offRT[myId-1];

        short* buf = ant_r + (size_t)myId * abR * 3;
        scR[myId] = extendOneDirection(seeds_r[myId], querySeg, dbSeg,
                                       qLen, tLen,
                                       EXTEND_RIGHTL, XDrop, buf);
      }
    }
  } // end target data (results copied back automatically)

  auto end_c = NOW;
  std::chrono::duration<double> compute = end_c - start_c;
  std::cout << "Device only time [seconds]:\t" << compute.count() << std::endl;

  // Combine left + right scores and update seeds
  for (int i = 0; i < numAlignments; i++) {
    int sl = scL[i], sr = scR[i];
    res[i] = (sl == UNDEF ? 0 : sl) + (sr == UNDEF ? 0 : sr) + kmer_length;
    setEndPositionH(seeds[i], getEndPositionH(seeds_r[i]));
    setEndPositionV(seeds[i], getEndPositionV(seeds_r[i]));
    std::cout << res[i] << "\n";
  }

  free(prefQ); free(prefT);
  free(suffQ); free(suffT);
  free(offLQ); free(offLT);
  free(offRQ); free(offRT);
  free(seeds_l); free(seeds_r);
  free(scL); free(scR);
  free(ant_l); free(ant_r);
}

// ---------------------------------------------------------------------------
// Top-level LOGAN entry point (matches original API)
// ---------------------------------------------------------------------------
std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) result.push_back(item);
  return result;
}

char basecomplement(char n) {
  switch (n) {
    case 'A': return 'T'; case 'T': return 'A';
    case 'G': return 'C'; case 'C': return 'G';
  }
  return 'N';
}

static constexpr int BATCH_SIZE = 30000;

void LOGAN(std::vector<std::vector<std::string>>& alignments,
           int ksize, int xdrop, int AlignmentsToBePerformed) {
  std::vector<int>    posV(AlignmentsToBePerformed);
  std::vector<int>    posH(AlignmentsToBePerformed);
  std::vector<SeedL>  seeds(AlignmentsToBePerformed);
  std::vector<std::string> seqsV(AlignmentsToBePerformed);
  std::vector<std::string> seqsH(AlignmentsToBePerformed);
  ScoringSchemeL sscheme(1, -1, -1, -1);

  for (int i = 0; i < AlignmentsToBePerformed; i++) {
    posV[i]  = std::stoi(alignments[i][1]);
    posH[i]  = std::stoi(alignments[i][3]);
    seqsV[i] = alignments[i][0];
    seqsH[i] = alignments[i][2];
    std::string strand = alignments[i][4];

    if (strand == "c") {
      std::transform(seqsH[i].begin(), seqsH[i].end(),
                     seqsH[i].begin(), basecomplement);
      posH[i] = (int)seqsH[i].size() - posH[i] - ksize;
    }
    SeedL_init(seeds[i], posH[i], posV[i], ksize);
  }

  int numLocal = BATCH_SIZE;
  for (int i = 0; i < AlignmentsToBePerformed; i += BATCH_SIZE) {
    if (AlignmentsToBePerformed < i + BATCH_SIZE)
      numLocal = AlignmentsToBePerformed - i;

    std::vector<std::string> tgt_b(seqsH.begin()+i, seqsH.begin()+i+numLocal);
    std::vector<std::string> qry_b(seqsV.begin()+i, seqsV.begin()+i+numLocal);
    std::vector<SeedL>       sds_b(seeds.begin()+i, seeds.begin()+i+numLocal);

    std::vector<int> resv(numLocal);
    extendSeedL_omp(sds_b, tgt_b, qry_b, xdrop, ksize,
                    resv.data(), numLocal);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <alignments_file> <kmer_size> <xdrop>\n";
    return 1;
  }

  std::ifstream input(argv[1]);
  if (!input.is_open()) {
    std::cerr << "Cannot open file: " << argv[1] << "\n";
    return 1;
  }

  int ksize = std::atoi(argv[2]);
  int xdrop = std::atoi(argv[3]);

  uint64_t AlignmentsToBePerformed =
    std::count(std::istreambuf_iterator<char>(input),
               std::istreambuf_iterator<char>(), '\n');
  input.seekg(0, std::ios_base::beg);

  std::vector<std::string> entries;
  for (uint64_t i = 0; i < AlignmentsToBePerformed; i++) {
    std::string line;
    std::getline(input, line);
    entries.push_back(line);
  }
  input.close();

  std::vector<std::vector<std::string>> alignments(AlignmentsToBePerformed);
  for (uint64_t i = 0; i < AlignmentsToBePerformed; i++)
    alignments[i] = split(entries[i], '\t');

  auto start = NOW;
  LOGAN(alignments, ksize, xdrop, (int)AlignmentsToBePerformed);
  auto end = NOW;
  std::chrono::duration<double> tot = end - start;
  std::cout << "Total execution time [seconds]:\t" << tot.count() << std::endl;

  return 0;
}

// OpenMP target offloading port of nbnxm (non-bonded neighbor list) benchmark.
// Computes Lennard-Jones + PME electrostatic interactions between atom clusters.
// Translated from Kokkos TeamPolicy version using flat parallel loop approach.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ============================================================
// Constants
// ============================================================
static constexpr int c_nbnxnGpuNumClusterPerSupercluster = 8;
static constexpr int c_clSize                            = 8;
static constexpr int c_nbnxnGpuClusterpairSplit          = 2;
static constexpr int c_nbnxnGpuJgroupSize                = (32 / c_nbnxnGpuNumClusterPerSupercluster);
static constexpr int c_nbnxnGpuExclSize                  = c_clSize * c_clSize / c_nbnxnGpuClusterpairSplit;
static constexpr int c_splitClSize                       = c_clSize / c_nbnxnGpuClusterpairSplit;
static constexpr int c_centralShiftIndex                 = 13;
static constexpr float c_nbnxnMinDistanceSquared         = 3.82e-07f;
static constexpr unsigned int NBNXN_INTERACTION_MASK_ALL = 0xffffffffU;
static constexpr unsigned superClInteractionMask         = ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);

static constexpr int grid_z   = 3199;
static constexpr int NUM_ATOMS = (grid_z * c_nbnxnGpuNumClusterPerSupercluster + c_clSize) * c_clSize + c_clSize;
static constexpr int NUM_CJ4   = 56881;
static constexpr int NUM_SCI   = 4806;
static constexpr int NUM_EXCL  = 19205;
static constexpr int NUM_NBFP  = 1024;
static constexpr int NUM_SHIFT = 45;
static constexpr int NUM_TYPES = 32;

// ============================================================
// Data structures
// ============================================================
struct nbnxn_im_ei_t {
    unsigned int imask   = 0U;
    int          excl_ind = 0;
};

struct nbnxn_cj4_t {
    int           cj[c_nbnxnGpuJgroupSize];
    nbnxn_im_ei_t imei[c_nbnxnGpuClusterpairSplit];
};

struct nbnxn_sci_t {
    int sci;
    int shift;
    int cj4_ind_start;
    int cj4_ind_end;
};

struct nbnxn_excl_t {
    unsigned int pair[c_nbnxnGpuExclSize];
};

struct Float4 { float x, y, z, w; };
struct Float3 { float x, y, z; };
struct Float2 { float x, y; };

// ============================================================
// PME correction polynomial
// ============================================================
#pragma omp declare target
static float pmeCorrF(float z2) {
    const float FN6 = -1.7357322914161492954e-8f;
    const float FN5 =  1.4703624142580877519e-6f;
    const float FN4 = -0.000053401640219807709149f;
    const float FN3 =  0.0010054721316683106153f;
    const float FN2 = -0.019278317264888380590f;
    const float FN1 =  0.069670166153766424023f;
    const float FN0 = -0.75225204789749321333f;
    const float FD4 = 0.0011193462567257629232f;
    const float FD3 = 0.014866955030185295499f;
    const float FD2 = 0.11583842382862377919f;
    const float FD1 = 0.50736591960530292870f;
    const float FD0 = 1.0f;

    const float z4      = z2 * z2;
    float polyFD0 = FD4 * z4 + FD2;
    const float polyFD1 = FD3 * z4 + FD1;
    polyFD0 = polyFD0 * z4 + FD0;
    polyFD0 = polyFD1 * z2 + polyFD0;
    polyFD0 = 1.0f / polyFD0;

    float polyFN0 = FN6 * z4 + FN4;
    float polyFN1 = FN5 * z4 + FN3;
    polyFN0 = polyFN0 * z4 + FN2;
    polyFN1 = polyFN1 * z4 + FN1;
    polyFN0 = polyFN0 * z4 + FN0;
    polyFN0 = polyFN1 * z2 + polyFN0;

    return polyFN0 * polyFD0;
}
#pragma omp end declare target

// ============================================================
// Kernel: flat loop over all (sci * clSize * clSize) threads
// For each supercluster (sci), we have team_size = clSize*clSize threads
// ============================================================
static void runNbnxmKernel(
    const Float4*       a_xq,
    const Float3*       a_shiftVec,
    const nbnxn_cj4_t*  a_cj4,
    const nbnxn_sci_t*  a_sci,
    const nbnxn_excl_t* a_excl,
    const int*          a_atomTypes,
    const Float2*       a_nbfp,
    int   numTypes,
    float rCoulombSq,
    float ewaldBeta,
    float epsFac,
    bool  calcShift,
    Float3* f_out,
    Float3* fShift_out,
    int num_sci)
{
    // Map to: outer loop over sci, inner loop over team threads (clSize*clSize)
    // We flatten: total = num_sci * clSize * clSize
    const int total = num_sci * c_clSize * c_clSize;
    const float beta2 = ewaldBeta * ewaldBeta;
    const float beta3 = ewaldBeta * ewaldBeta * ewaldBeta;
    const int prunedClusterPairSize = c_clSize * c_splitClSize;

    #pragma omp target teams distribute parallel for thread_limit(256) \
        map(to: a_xq[0:NUM_ATOMS], a_shiftVec[0:NUM_SHIFT], \
                a_cj4[0:NUM_CJ4], a_sci[0:num_sci], a_excl[0:NUM_EXCL], \
                a_atomTypes[0:NUM_ATOMS], a_nbfp[0:NUM_NBFP]) \
        map(tofrom: f_out[0:NUM_ATOMS], fShift_out[0:NUM_SHIFT])
    for (int flat = 0; flat < total; flat++) {
        const int bidx  = flat / (c_clSize * c_clSize);
        const int tidx  = flat % (c_clSize * c_clSize);
        const int tidxi = tidx % c_clSize;
        const int tidxj = tidx / c_clSize;

        const nbnxn_sci_t nbSci     = a_sci[bidx];
        const int         sci       = nbSci.sci;
        const int         cij4Start = nbSci.cj4_ind_start;
        const int         cij4End   = nbSci.cj4_ind_end;
        const int         sciShift  = nbSci.shift;

        // Load i-atom position (use local variables)
        const int ci  = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj;
        const int ai  = ci * c_clSize + tidxi;
        Float4 xqi    = a_xq[ai];
        const Float3 shiftV = a_shiftVec[sciShift];
        xqi.x += shiftV.x; xqi.y += shiftV.y; xqi.z += shiftV.z;
        xqi.w *= epsFac;
        (void)a_atomTypes[ai]; // atomTypeI used via ai_i in inner loop

        const int imeiIdx = tidx / prunedClusterPairSize;

        float fCiBufX[c_nbnxnGpuNumClusterPerSupercluster] = {};
        float fCiBufY[c_nbnxnGpuNumClusterPerSupercluster] = {};
        float fCiBufZ[c_nbnxnGpuNumClusterPerSupercluster] = {};

        const bool nonSelfInteraction =
            !(sciShift == c_centralShiftIndex && tidxj <= tidxi);

        for (int j4 = cij4Start; j4 < cij4End; ++j4) {
            unsigned imask = a_cj4[j4].imei[imeiIdx].imask;
            if (!imask) continue;

            const int      wexclIdx = a_cj4[j4].imei[imeiIdx].excl_ind;
            const unsigned wexcl    = a_excl[wexclIdx].pair[tidx & (prunedClusterPairSize - 1)];

            for (int jm = 0; jm < c_nbnxnGpuJgroupSize; ++jm) {
                const bool maskSet = (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster))) != 0;
                if (!maskSet) continue;

                unsigned  maskJI = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
                const int cj     = a_cj4[j4].cj[jm];
                const int aj     = cj * c_clSize + tidxj;

                const Float4 xqj = a_xq[aj];
                const float  xj  = xqj.x, yj = xqj.y, zj = xqj.z;
                const float  qj  = xqj.w;
                const int atomTypeJ = a_atomTypes[aj];

                float fCjX = 0.f, fCjY = 0.f, fCjZ = 0.f;

                for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; ++i) {
                    if (imask & maskJI) {
                        // Use per-thread i atom (tidxi) for i-cluster atom
                        const int ci_i = sci * c_nbnxnGpuNumClusterPerSupercluster + i;
                        const int ai_i = ci_i * c_clSize + tidxi;
                        const Float4 xqi_i = { a_xq[ai_i].x + shiftV.x,
                                               a_xq[ai_i].y + shiftV.y,
                                               a_xq[ai_i].z + shiftV.z,
                                               a_xq[ai_i].w * epsFac };
                        const float xi = xqi_i.x, yi = xqi_i.y, zi = xqi_i.z;

                        const float rvx = xi - xj;
                        const float rvy = yi - yj;
                        const float rvz = zi - zj;
                        float        r2 = rvx*rvx + rvy*rvy + rvz*rvz;

                        const float pairExclMask = (wexcl & maskJI) ? 1.0f : 0.0f;
                        const bool  notExcluded  = (nonSelfInteraction | (ci_i != cj));

                        if ((r2 < rCoulombSq) && notExcluded) {
                            const float qi = xqi_i.w;
                            const int atI  = a_atomTypes[ai_i];
                            const Float2 c6c12 = a_nbfp[numTypes * atI + atomTypeJ];
                            const float  c6    = c6c12.x;
                            const float  c12   = c6c12.y;

                            if (r2 < c_nbnxnMinDistanceSquared) r2 = c_nbnxnMinDistanceSquared;

                            const float rInv  = 1.0f / sqrtf(r2);
                            const float r2Inv = rInv * rInv;
                            float r6Inv = r2Inv * r2Inv * r2Inv;
                            r6Inv *= pairExclMask;
                            float fInvR = r6Inv * (c12 * r6Inv - c6) * r2Inv;
                            fInvR += qi * qj * (pairExclMask * r2Inv * rInv + pmeCorrF(beta2 * r2) * beta3);

                            const float fx = rvx * fInvR;
                            const float fy = rvy * fInvR;
                            const float fz = rvz * fInvR;

                            fCjX -= fx; fCjY -= fy; fCjZ -= fz;
                            fCiBufX[i] += fx;
                            fCiBufY[i] += fy;
                            fCiBufZ[i] += fz;
                        }
                    }
                    maskJI += maskJI;
                }

                // Reduce j-forces atomically
                #pragma omp atomic update
                f_out[aj].x += fCjX;
                #pragma omp atomic update
                f_out[aj].y += fCjY;
                #pragma omp atomic update
                f_out[aj].z += fCjZ;
            }
        }

        // Reduce i-forces atomically
        for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; ++i) {
            const int aidx = (sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi;
            #pragma omp atomic update
            f_out[aidx].x += fCiBufX[i];
            #pragma omp atomic update
            f_out[aidx].y += fCiBufY[i];
            #pragma omp atomic update
            f_out[aidx].z += fCiBufZ[i];

            if (calcShift && sciShift != c_centralShiftIndex) {
                #pragma omp atomic update
                fShift_out[sciShift].x += fCiBufX[i];
                #pragma omp atomic update
                fShift_out[sciShift].y += fCiBufY[i];
                #pragma omp atomic update
                fShift_out[sciShift].z += fCiBufZ[i];
            }
        }
    }
}

// ============================================================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    // Allocate host arrays
    Float4*       a_xq       = (Float4*)      malloc(NUM_ATOMS * sizeof(Float4));
    Float3*       a_f_base   = (Float3*)      malloc(NUM_ATOMS * sizeof(Float3));
    Float3*       shiftVec   = (Float3*)      malloc(NUM_SHIFT * sizeof(Float3));
    Float3*       fShift_base= (Float3*)      malloc(NUM_SHIFT * sizeof(Float3));
    nbnxn_cj4_t*  cj4        = (nbnxn_cj4_t*)malloc(NUM_CJ4   * sizeof(nbnxn_cj4_t));
    nbnxn_sci_t*  sci        = (nbnxn_sci_t*)malloc(NUM_SCI   * sizeof(nbnxn_sci_t));
    nbnxn_excl_t* excl       = (nbnxn_excl_t*)malloc(NUM_EXCL  * sizeof(nbnxn_excl_t));
    int*          atomTypes  = (int*)         malloc(NUM_ATOMS * sizeof(int));
    Float2*       nbfp       = (Float2*)      malloc(NUM_NBFP  * sizeof(Float2));
    Float3*       f_out      = (Float3*)      malloc(NUM_ATOMS * sizeof(Float3));
    Float3*       fShift_out = (Float3*)      malloc(NUM_SHIFT * sizeof(Float3));

    // Initialize
    for (int i = 0; i < NUM_ATOMS;  ++i) a_xq[i]       = {1.0f, 0.5f, 0.25f, 0.125f};
    for (int i = 0; i < NUM_ATOMS;  ++i) a_f_base[i]   = {1.0f, 0.5f, 0.25f};
    for (int i = 0; i < NUM_SHIFT;  ++i) shiftVec[i]   = {1.0f, 0.5f, 0.25f};
    for (int i = 0; i < NUM_SHIFT;  ++i) fShift_base[i]= {1.0f, 0.5f, 0.25f};
    for (int i = 0; i < NUM_CJ4;    ++i) {
        for (int j = 0; j < c_nbnxnGpuJgroupSize; ++j) cj4[i].cj[j] = j + i;
        for (int j = 0; j < c_nbnxnGpuClusterpairSplit; ++j) {
            cj4[i].imei[j].imask = 0U;
            cj4[i].imei[j].excl_ind = 0;
        }
    }
    for (int i = 0; i < NUM_SCI;    ++i) sci[i]        = {i, 0, 8*i, 8*i+7};
    for (int i = 0; i < NUM_EXCL;   ++i) for (int j=0;j<c_nbnxnGpuExclSize;++j) excl[i].pair[j]=7;
    for (int i = 0; i < NUM_ATOMS;  ++i) atomTypes[i]  = (i % 2);
    for (int i = 0; i < NUM_NBFP;   ++i) nbfp[i]       = {0.5f, 0.25f};

    const float rCoulombSq = 1.0f;
    const float ewaldBeta  = 3.12341f;
    const float epsFac     = 138.935f;

    // Warm-up
    memcpy(f_out,      a_f_base,    NUM_ATOMS * sizeof(Float3));
    memcpy(fShift_out, fShift_base, NUM_SHIFT * sizeof(Float3));
    runNbnxmKernel(a_xq, shiftVec, cj4, sci, excl, atomTypes, nbfp,
                   NUM_TYPES, rCoulombSq, ewaldBeta, epsFac, false,
                   f_out, fShift_out, NUM_SCI);

    // Benchmark without shift
    memcpy(f_out,      a_f_base,    NUM_ATOMS * sizeof(Float3));
    memcpy(fShift_out, fShift_base, NUM_SHIFT * sizeof(Float3));

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
        runNbnxmKernel(a_xq, shiftVec, cj4, sci, excl, atomTypes, nbfp,
                       NUM_TYPES, rCoulombSq, ewaldBeta, epsFac, false,
                       f_out, fShift_out, NUM_SCI);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time (w/o shift): %f (us)\n", (ns * 1e-3) / repeat);

    // Benchmark with shift
    memcpy(f_out,      a_f_base,    NUM_ATOMS * sizeof(Float3));
    memcpy(fShift_out, fShift_base, NUM_SHIFT * sizeof(Float3));

    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
        runNbnxmKernel(a_xq, shiftVec, cj4, sci, excl, atomTypes, nbfp,
                       NUM_TYPES, rCoulombSq, ewaldBeta, epsFac, true,
                       f_out, fShift_out, NUM_SCI);
    }
    t1 = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time (w/ shift): %f (us)\n", (ns * 1e-3) / repeat);

    free(a_xq); free(a_f_base); free(shiftVec); free(fShift_base);
    free(cj4); free(sci); free(excl); free(atomTypes); free(nbfp);
    free(f_out); free(fShift_out);
    return 0;
}

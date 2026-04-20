// OpenMP target port of logic-rewrite-kokkos: AIG cut enumeration.

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

static constexpr int CUT_SET_SIZE = 8;
static constexpr int MAX_LEAVES   = 4;

#define ID(i, j) ((i) * CUT_SET_SIZE + (j))

struct Cut {
    int  sign;
    int  leaves[MAX_LEAVES];
    unsigned truthtable : 16;
    unsigned value      : 11;
    unsigned nLeaves    : 4;
    bool     used;
};

#pragma omp declare target

inline int CutFindValue(const Cut* cut, const int* nRef) {
    int value = 0, nOnes = 0;
    for (int i = 0; i < (int)cut->nLeaves; i++) {
        value += nRef[cut->leaves[i]];
        nOnes += (nRef[cut->leaves[i]] == 1);
    }
    if (cut->nLeaves < 2) return 1001;
    if (value > 1000) value = 1000;
    if (nOnes > 3) value = 5 - nOnes;
    return value;
}

inline int CountOnes(unsigned uWord) {
    uWord = (uWord & 0x55555555u) + ((uWord >> 1) & 0x55555555u);
    uWord = (uWord & 0x33333333u) + ((uWord >> 2) & 0x33333333u);
    uWord = (uWord & 0x0F0F0F0Fu) + ((uWord >> 4) & 0x0F0F0F0Fu);
    uWord = (uWord & 0x00FF00FFu) + ((uWord >> 8) & 0x00FF00FFu);
    return (int)((uWord & 0x0000FFFFu) + (uWord >> 16));
}

inline int FindCut(int idx, Cut* cuts) {
    for (int i = 0; i < CUT_SET_SIZE; i++)
        if (!cuts[ID(idx, i)].used) return i;
    int ans = -1;
    for (int i = 0; i < CUT_SET_SIZE; i++)
        if (cuts[ID(idx, i)].nLeaves > 2) {
            if (ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                ans = i;
        }
    if (ans == -1)
        for (int i = 0; i < CUT_SET_SIZE; i++)
            if (cuts[ID(idx, i)].nLeaves == 2) {
                if (ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    if (ans == -1)
        for (int i = 0; i < CUT_SET_SIZE; i++)
            if (cuts[ID(idx, i)].nLeaves < 2) {
                if (ans == -1 || cuts[ID(idx, i)].value < cuts[ID(idx, ans)].value)
                    ans = i;
            }
    cuts[ID(idx, ans)].used = false;
    return ans;
}

inline int MergeCutOrdered(Cut a, Cut b, Cut* out) {
    int c = 0, i = 0, k = 0;
    for (c = 0; c < MAX_LEAVES; c++) {
        if (k == (int)b.nLeaves) {
            if (i == (int)a.nLeaves) { out->nLeaves = c; return 1; }
            out->leaves[c] = a.leaves[i++]; continue;
        }
        if (i == (int)a.nLeaves) {
            out->leaves[c] = b.leaves[k++]; continue;
        }
        if (a.leaves[i] < b.leaves[k]) { out->leaves[c] = a.leaves[i++]; continue; }
        if (a.leaves[i] > b.leaves[k]) { out->leaves[c] = b.leaves[k++]; continue; }
        out->leaves[c] = a.leaves[i++]; k++;
    }
    if (i < (int)a.nLeaves || k < (int)b.nLeaves) return 0;
    out->nLeaves = c;
    return 1;
}

inline int MergeCut(Cut a, Cut b, Cut* out) {
    if (a.nLeaves >= b.nLeaves) {
        if (!MergeCutOrdered(a, b, out)) return 0;
    } else {
        if (!MergeCutOrdered(b, a, out)) return 0;
    }
    out->sign = a.sign | b.sign;
    out->used = true;
    return 1;
}

inline int CutDominance(Cut a, Cut b) {
    for (int i = 0; i < (int)a.nLeaves; i++) {
        int ok = 0;
        for (int j = 0; j < (int)b.nLeaves; j++)
            if (b.leaves[j] == a.leaves[i]) ok = 1;
        if (!ok) return 0;
    }
    return 1;
}

inline int CutFilter(Cut* cut, int id) {
    for (int i = 0; i < CUT_SET_SIZE; i++) if (i != id && cut[i].used) {
        if (cut[i].nLeaves > cut[id].nLeaves) {
            if ((cut[i].sign & cut[id].sign) != cut[id].sign) continue;
            if (CutDominance(cut[id], cut[i])) cut[i].used = false;
        } else {
            if ((cut[i].sign & cut[id].sign) != cut[i].sign) continue;
            if (CutDominance(cut[i], cut[id])) { cut[id].used = false; return 1; }
        }
    }
    return 0;
}

inline int CutTruthPhase(Cut x, Cut cut) {
    int phase = 0;
    for (int i = 0; i < (int)cut.nLeaves; i++)
        for (int j = 0; j < (int)x.nLeaves; j++)
            if (x.leaves[j] == cut.leaves[i]) phase |= 1 << i;
    return phase;
}

inline int CutTruthSwapAdjacentVars(int uTruth, int iVar) {
    if (iVar == 0) return (uTruth & 0x99999999) | ((uTruth & 0x22222222) << 1) | ((uTruth & 0x44444444) >> 1);
    if (iVar == 1) return (uTruth & 0xC3C3C3C3) | ((uTruth & 0x0C0C0C0C) << 2) | ((uTruth & 0x30303030) >> 2);
    if (iVar == 2) return (uTruth & 0xF00FF00F) | ((uTruth & 0x00F000F0) << 4) | ((uTruth & 0x0F000F00) >> 4);
    return 0;
}

inline int CutTruthStretch(int tt, int nVar, int phase) {
    int Var = nVar - 1;
    for (int i = 3; i >= 0; i--) if (phase >> i & 1) {
        for (int k = Var; k < i; k++) tt = CutTruthSwapAdjacentVars(tt, k);
        Var--;
    }
    return tt;
}

inline int CutTruthtable(Cut cut, Cut a, Cut b, int aC, int bC) {
    int tt0 = aC ? ~a.truthtable : a.truthtable;
    int tt1 = bC ? ~b.truthtable : b.truthtable;
    tt0 = CutTruthStretch(tt0, a.nLeaves, CutTruthPhase(a, cut));
    tt1 = CutTruthStretch(tt1, b.nLeaves, CutTruthPhase(b, cut));
    return tt0 & tt1;
}

inline int CutTruthShrink(int uTruth, int nVars, int Phase) {
    int Var = 0;
    for (int i = 0; i < 4; i++) if (Phase & (1 << i)) {
        for (int k = i - 1; k >= Var; k--) uTruth = CutTruthSwapAdjacentVars(uTruth, k);
        Var++;
    }
    return uTruth;
}

inline int MinimizeCutSupport(Cut* cut) {
    const int masks[4][2] = {
        {0x5555, 0xAAAA}, {0x3333, 0xCCCC}, {0x0F0F, 0xF0F0}, {0x00FF, 0xFF00}
    };
    int phase = 0, truth = cut->truthtable & 0xFFFF, nLeaves = cut->nLeaves;
    for (int i = 0; i < (int)cut->nLeaves; i++)
        if ((truth & masks[i][0]) == ((truth & masks[i][1]) >> (1 << i)))
            nLeaves--;
        else
            phase |= 1 << i;
    if (nLeaves == (int)cut->nLeaves) return 0;
    truth = CutTruthShrink(truth, cut->nLeaves, phase);
    cut->truthtable = truth & 0xFFFF;
    cut->sign  = 0;
    int k = 0;
    for (int i = 0; i < (int)cut->nLeaves; i++) if (phase >> i & 1) {
        cut->leaves[k++] = cut->leaves[i];
        cut->sign |= 1 << (31 & cut->leaves[i]);
    }
    cut->nLeaves = nLeaves;
    return 1;
}

#pragma omp end declare target

static void buildRandomAIG(int nPIs, int nNodes,
    std::vector<int>& fanin0, std::vector<int>& fanin1,
    std::vector<int>& isC0,   std::vector<int>& isC1,
    std::vector<int>& level,  std::vector<int>& nRef)
{
    // Node 0 = constant; nodes 1..nPIs = primary inputs
    int total = 1 + nPIs + nNodes;
    fanin0.assign(total, 0);
    fanin1.assign(total, 0);
    isC0  .assign(total, 0);
    isC1  .assign(total, 0);
    level .assign(total, 0);
    nRef  .assign(total, 0);

    std::mt19937 rng(42);

    for (int i = 1 + nPIs; i < total; i++) {
        int f0 = std::uniform_int_distribution<int>(1, i - 1)(rng);
        int f1 = std::uniform_int_distribution<int>(1, i - 1)(rng);
        int c0 = std::uniform_int_distribution<int>(0, 1)(rng);
        int c1 = std::uniform_int_distribution<int>(0, 1)(rng);
        fanin0[i] = f0 * 2 + c0;
        fanin1[i] = f1 * 2 + c1;
        isC0  [i] = c0;
        isC1  [i] = c1;
        int nodeId = i;
        int f0id   = fanin0[nodeId] >> 1;
        int f1id   = fanin1[nodeId] >> 1;
        level [nodeId] = 1 + std::max(level[f0id], level[f1id]);
        nRef  [f0id]++;
        nRef  [f1id]++;
    }
}

int main(int argc, char* argv[]) {
    int nPIs   = 64;
    int nNodes = 1024;
    if (argc > 1) nNodes = atoi(argv[1]);

    printf("AIG rewrite cut enumeration: %d PIs, %d AND nodes\n", nPIs, nNodes);

    std::vector<int> fanin0, fanin1, isC0_h, isC1_h, level_h, nRef_h;
    buildRandomAIG(nPIs, nNodes, fanin0, fanin1, isC0_h, isC1_h, level_h, nRef_h);

    int nObjs = (int)fanin0.size();

    int maxLevel = *std::max_element(level_h.begin(), level_h.end());
    std::vector<std::vector<int>> levelNodes(maxLevel + 1);
    for (int i = 1 + nPIs; i < nObjs; i++)
        levelNodes[level_h[i]].push_back(i);

    // Host arrays
    int* h_fanin0 = fanin0.data();
    int* h_fanin1 = fanin1.data();
    int* h_isC0   = isC0_h.data();
    int* h_isC1   = isC1_h.data();
    int* h_nRef   = nRef_h.data();
    Cut* h_cuts   = new Cut[nObjs * CUT_SET_SIZE]();

    // Device allocations
    int* d_fanin0 = (int*)malloc(nObjs * sizeof(int));
    int* d_fanin1 = (int*)malloc(nObjs * sizeof(int));
    int* d_isC0   = (int*)malloc(nObjs * sizeof(int));
    int* d_isC1   = (int*)malloc(nObjs * sizeof(int));
    int* d_nRef   = (int*)malloc(nObjs * sizeof(int));
    Cut* d_cuts   = (Cut*)malloc((size_t)nObjs * CUT_SET_SIZE * sizeof(Cut));

    memcpy(d_fanin0, h_fanin0, nObjs * sizeof(int));
    memcpy(d_fanin1, h_fanin1, nObjs * sizeof(int));
    memcpy(d_isC0,   h_isC0,   nObjs * sizeof(int));
    memcpy(d_isC1,   h_isC1,   nObjs * sizeof(int));
    memcpy(d_nRef,   h_nRef,   nObjs * sizeof(int));
    memset(d_cuts, 0, (size_t)nObjs * CUT_SET_SIZE * sizeof(Cut));

    size_t cuts_sz = (size_t)nObjs * CUT_SET_SIZE;

    #pragma omp target enter data \
        map(to: d_fanin0[0:nObjs], d_fanin1[0:nObjs], \
                d_isC0[0:nObjs], d_isC1[0:nObjs], d_nRef[0:nObjs]) \
        map(alloc: d_cuts[0:cuts_sz])

    #pragma omp target update to(d_cuts[0:cuts_sz])

    // Phase 1: assign trivial cuts to PIs
    #pragma omp target teams distribute parallel for thread_limit(256) \
        map(tofrom: d_cuts[0:cuts_sz])
    for (int idx = 1; idx <= nPIs; idx++) {
        Cut* cut       = &d_cuts[ID(idx, 0)];
        cut->used      = true;
        cut->sign      = 1 << (idx & 31);
        cut->truthtable = 0xAAAA;
        cut->nLeaves   = 1;
        cut->leaves[0] = idx;
        cut->value     = (unsigned)CutFindValue(cut, d_nRef);
    }

    printf("Enumerating cuts over %d levels ...\n", maxLevel);

    auto t0 = std::chrono::steady_clock::now();

    // Phase 2: level-by-level cut enumeration
    for (int lev = 1; lev <= maxLevel; lev++) {
        const std::vector<int>& nodes = levelNodes[lev];
        if (nodes.empty()) continue;

        int delta = nodes[0] - 1;
        int cnt   = (int)nodes.size();

        #pragma omp target teams distribute parallel for thread_limit(256) \
            map(tofrom: d_cuts[0:cuts_sz])
        for (int tidx = 0; tidx < cnt; tidx++) {
            int idx = tidx + delta + 1;

            // Trivial cut
            Cut* cut        = &d_cuts[ID(idx, 0)];
            cut->used       = true;
            cut->sign       = 1 << (idx & 31);
            cut->truthtable = 0xAAAA;
            cut->nLeaves    = 1;
            cut->leaves[0]  = idx;
            cut->value      = (unsigned)CutFindValue(cut, d_nRef);

            int in0 = d_fanin0[idx] >> 1;
            int in1 = d_fanin1[idx] >> 1;

            for (int i = 0; i < CUT_SET_SIZE; i++) {
                if (!d_cuts[ID(in0, i)].used) continue;
                for (int j = 0; j < CUT_SET_SIZE; j++) {
                    if (!d_cuts[ID(in1, j)].used) continue;

                    Cut a = d_cuts[ID(in0, i)];
                    Cut b = d_cuts[ID(in1, j)];

                    if (CountOnes((unsigned)(a.sign | b.sign)) > MAX_LEAVES) continue;

                    int cutId = FindCut(idx, d_cuts);
                    Cut newCut;
                    if (!MergeCut(a, b, &newCut)) continue;
                    d_cuts[ID(idx, cutId)] = newCut;

                    if (CutFilter(&d_cuts[ID(idx, 0)], cutId)) continue;

                    d_cuts[ID(idx, cutId)].truthtable = (unsigned)(
                        0xFFFF & CutTruthtable(d_cuts[ID(idx, cutId)], a, b,
                                              d_isC0[idx], d_isC1[idx]));

                    if (MinimizeCutSupport(&d_cuts[ID(idx, cutId)]))
                        CutFilter(&d_cuts[ID(idx, 0)], cutId);

                    d_cuts[ID(idx, cutId)].value =
                        (unsigned)CutFindValue(&d_cuts[ID(idx, cutId)], d_nRef);

                    if (d_cuts[ID(idx, cutId)].nLeaves < 2) goto done;
                }
            }
            done:;
        }
    }

    #pragma omp target update from(d_cuts[0:cuts_sz])

    auto t1 = std::chrono::steady_clock::now();
    printf("Cut enumeration time: %.6f (s)\n",
           std::chrono::duration<double>(t1 - t0).count());

    long long totalCuts = 0;
    for (int i = 0; i < nObjs; i++)
        for (int j = 0; j < CUT_SET_SIZE; j++)
            if (d_cuts[ID(i, j)].used) ++totalCuts;
    printf("Total valid cuts: %lld\n", totalCuts);

    #pragma omp target exit data \
        map(delete: d_fanin0[0:nObjs], d_fanin1[0:nObjs], \
                    d_isC0[0:nObjs], d_isC1[0:nObjs], d_nRef[0:nObjs], \
                    d_cuts[0:cuts_sz])

    free(d_fanin0); free(d_fanin1);
    free(d_isC0); free(d_isC1); free(d_nRef); free(d_cuts);
    delete[] h_cuts;
    return 0;
}

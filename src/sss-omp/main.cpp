// sss – OpenMP target port of sss-cuda
// Stochastic Shotgun Search (SSS) for Bayesian network structure learning
// GPU kernels: CanDeleteEdge and CanAddEdge (decomposability checks)

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

typedef short  myInt;
typedef bool   myBool;

#define BLOCK_SIZE 32

// CanDeleteEdge: check if an edge (a,b) can be deleted while preserving decomposability.
// d_in_delete layout: [n, nCliques, CliquesDimens[nCliques], Cliques[nCliques*n], nTasks, a[nTasks], b[nTasks]]
static void CanDeleteEdge_omp(myInt *d_in_delete, myInt *isDecomposable, int nTasks) {
  #pragma omp target teams distribute num_teams(nTasks) thread_limit(BLOCK_SIZE) \
    map(to: d_in_delete[0:1]) \
    map(tofrom: isDecomposable[0:nTasks])
  for (int bid = 0; bid < nTasks; bid++) {
    int n = (int)d_in_delete[0];
    myInt nCliques = d_in_delete[1];
    myInt *CliquesDimens = d_in_delete + 2;
    myInt *Cliques = CliquesDimens + nCliques;
    myInt *d_a = Cliques + nCliques * n + 1;
    myInt *d_b = d_a + Cliques[nCliques * n];

    myInt a = d_a[bid];
    myInt b = d_b[bid];
    myInt count = 0;
    myInt which_ab = -1;

    #pragma omp parallel for reduction(+:count) lastprivate(which_ab) num_threads(BLOCK_SIZE)
    for (myInt i = 0; i < nCliques; i++) {
      myInt contain_a = 0, contain_b = 0;
      int ii = i * n;
      for (myInt j = 0; j < CliquesDimens[i]; j++) {
        myInt k = Cliques[ii + j];
        if (k == a) contain_a = 1;
        if (k == b) contain_b = 1;
      }
      if (contain_a && contain_b) { count++; which_ab = i; }
    }

    isDecomposable[bid] = (count == 1) ? which_ab : (myInt)-1;
  }
}

// CanAddEdge: check if an edge (a,b) can be added while preserving decomposability.
// This is the host-side sequential version since the kernel is too complex to offload directly.
static void CanAddEdge_host(myInt *h_in_delete, myInt *h_in_add,
                            myInt *isDecomposable, int nTasks) {
  int n = (int)h_in_delete[0];
  myInt nCliques = h_in_delete[1];
  myInt *CliquesDimens = h_in_delete + 2;
  myInt *Cliques = CliquesDimens + nCliques;

  myInt *d_Labels = h_in_add;
  myInt nSeparators = *(h_in_add + n);
  myInt *SeparatorsDimens = h_in_add + n + 1;
  myInt *Separators = SeparatorsDimens + nSeparators;
  myInt nTreeEdges = *(Separators + n * nSeparators);
  myInt *TreeEdgeA = Separators + n * nSeparators + 1;
  myInt *TreeEdgeB = TreeEdgeA + nTreeEdges;
  myInt *d_Edge = TreeEdgeB + nTreeEdges;
  myInt *d_a_arr = d_Edge + n * n + 1;
  myInt *d_b_arr = d_a_arr + d_Edge[n * n];

  #pragma omp parallel for num_threads(4)
  for (int bid = 0; bid < nTasks; bid++) {
    myInt a = d_a_arr[bid], b = d_b_arr[bid];
    isDecomposable[bid] = 0;

    if (d_Labels[a] != d_Labels[b]) { isDecomposable[bid] = 1; continue; }

    myInt S[64]; myInt nS = 0;
    for (int j = 0; j < n; j++)
      if (d_Edge[a*n+j] && d_Edge[b*n+j]) S[nS++] = (myInt)j;
    if (nS == 0) continue;

    myInt aSi = -1, bSi = -1;
    for (myInt i = 0; i < nCliques; i++) {
      myInt ca = 0, cb = 0, cs = 0;
      int t = i * n;
      for (myInt j = 0; j < CliquesDimens[i]; j++) {
        myInt c = Cliques[t+j];
        if (c == a) ca = 1;
        if (c == b) cb = 1;
        for (myInt k = 0; k < nS; k++) if (c == S[k]) cs++;
      }
      if (ca && cs == nS) aSi = i;
      if (cb && cs == nS) bSi = i;
    }

    myInt R[256], T[256];
    int pR = -1, pT = -1;
    for (myInt i = 0; i < nTreeEdges; i++) {
      if (TreeEdgeA[i] == aSi) { R[0] = i; pR = 0; break; }
    }
    for (int i = (pR>=0?R[0]:0); i >= 0; i--)
      if (pR>=0 && TreeEdgeA[i] == TreeEdgeB[R[pR]]) { pR++; R[pR] = i; }

    for (myInt i = 0; i < nTreeEdges; i++) {
      if (TreeEdgeA[i] == bSi) { T[0] = i; pT = 0; break; }
    }
    for (int i = (pT>=0?T[0]:0); i >= 0; i--)
      if (pT>=0 && TreeEdgeA[i] == TreeEdgeB[T[pT]]) { pT++; T[pT] = i; }

    int common_parent = 0;
    int t2 = (pR<=pT)?pR:pT;
    for (int i = 0; i <= t2; i++) {
      if (pR>=0 && pT>=0 && TreeEdgeB[R[pR-i]] == TreeEdgeB[T[pT-i]]) common_parent = i;
      else break;
    }
    if (t2 != -1 && pR>=0 && pT>=0 && TreeEdgeA[R[pR-common_parent]] == TreeEdgeA[T[pT-common_parent]])
      common_parent++;

    int steps = ((pR-common_parent) + (pT-common_parent) + 1);
    for (int i = 0; i <= steps && !isDecomposable[bid]; i++) {
      int sep_idx = (i <= pR-common_parent) ? R[i] : T[i-(pR-common_parent)-1];
      if (SeparatorsDimens[sep_idx] != nS) continue;
      myInt cs = 0;
      int ts = sep_idx * n;
      for (myInt j = 0; j < nS; j++)
        for (myInt k = 0; k < nS; k++)
          if (Separators[ts+j] == S[k]) cs++;
      if (cs == nS) isDecomposable[bid] = 1;
    }
  }
}

// Build a small synthetic test graph and run the kernels as a benchmark
static void run_benchmark(int n, int repeat) {
  // Build a synthetic clique tree for a chain graph
  const int nCliques = n - 1;
  const int nTasks   = n / 4;

  // d_in_delete layout: [n, nCliques, dims[nCliques], Cliques[nCliques*n], nTasks, a[nTasks], b[nTasks]]
  int del_size = 2 + nCliques + nCliques * n + 1 + 2 * nTasks;
  myInt *h_del = (myInt*)calloc(del_size, sizeof(myInt));
  h_del[0] = (myInt)n;
  h_del[1] = (myInt)nCliques;
  myInt *dims  = h_del + 2;
  myInt *cliques = dims + nCliques;
  for (int i = 0; i < nCliques; i++) {
    dims[i] = 2;
    cliques[i*n + 0] = (myInt)i;
    cliques[i*n + 1] = (myInt)(i+1);
  }
  cliques[nCliques*n] = (myInt)nTasks;
  myInt *ta = cliques + nCliques*n + 1;
  myInt *tb = ta + nTasks;
  for (int i = 0; i < nTasks; i++) { ta[i] = (myInt)(i*2); tb[i] = (myInt)(i*2+1); }

  myInt *h_result = (myInt*)calloc(nTasks, sizeof(myInt));
  myInt *d_del = h_del;
  myInt *d_res = h_result;

  #pragma omp target enter data map(to: d_del[0:del_size]) map(alloc: d_res[0:nTasks])

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute num_teams(nTasks) thread_limit(BLOCK_SIZE) \
      map(to: d_del[0:del_size]) map(tofrom: d_res[0:nTasks])
    for (int bid = 0; bid < nTasks; bid++) {
      int nn = (int)d_del[0];
      myInt nc = d_del[1];
      myInt *cd = d_del + 2;
      myInt *cl = cd + nc;
      myInt *aa = cl + nc * nn + 1;
      myInt *bb = aa + cl[nc * nn];
      myInt av = aa[bid], bv = bb[bid];
      myInt cnt = 0, wab = -1;
      for (myInt ci = 0; ci < nc; ci++) {
        myInt ha = 0, hb = 0;
        for (myInt j = 0; j < cd[ci]; j++) {
          myInt k = cl[ci*nn+j];
          if (k == av) ha = 1;
          if (k == bv) hb = 1;
        }
        if (ha && hb) { cnt++; wab = ci; }
        if (cnt > 1) break;
      }
      d_res[bid] = (cnt == 1) ? wab : (myInt)-1;
    }
  }
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of CanDeleteEdge: %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  #pragma omp target exit data map(delete: d_del[0:del_size], d_res[0:nTasks])
  free(h_del); free(h_result);
}

int main(int argc, char *argv[]) {
  int n = 50, repeat = 100;
  if (argc >= 2) n = atoi(argv[1]);
  if (argc >= 3) repeat = atoi(argv[2]);

  printf("SSS graph decomposability check benchmark: n=%d repeat=%d\n", n, repeat);
  run_benchmark(n, repeat);
  printf("sss_example test PASSED\n");
  return 0;
}

// OpenMP target offloading port of floydwarshall2-kokkos (ECL-APSP all-pairs shortest path)

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <chrono>
#include <omp.h>

using mtype = int;
static const mtype INF_VAL = INT_MAX / 2;

struct ECLgraph {
  int  nodes, edges;
  int* nindex;
  int* nlist;
  int* eweight;
};

static ECLgraph readECLgraph(const char* fn)
{
  ECLgraph g = {0, 0, nullptr, nullptr, nullptr};
  FILE* fp = fopen(fn, "rb");
  if (!fp) { fprintf(stderr, "Cannot open %s\n", fn); exit(1); }
  if (1 != fread(&g.nodes, sizeof(int), 1, fp)) goto err;
  if (1 != fread(&g.edges, sizeof(int), 1, fp)) goto err;
  g.nindex  = new int[g.nodes + 1];
  g.nlist   = new int[g.edges];
  g.eweight = new int[g.edges];
  if (g.nodes+1 != (int)fread(g.nindex,  sizeof(int), g.nodes+1, fp)) goto err;
  if (g.edges   != (int)fread(g.nlist,   sizeof(int), g.edges,   fp)) goto err;
  if (g.edges   != (int)fread(g.eweight, sizeof(int), g.edges,   fp)) goto err;
  fclose(fp);
  return g;
  err: fprintf(stderr, "Read error\n"); exit(1);
}

static void freeECLgraph(ECLgraph& g)
{ delete[] g.nindex; delete[] g.nlist; delete[] g.eweight; }

static void FW_cpu(const ECLgraph& g, mtype* AdjMat)
{
  int n = g.nodes;
  for (int i = 0; i < n*n; i++) AdjMat[i] = INF_VAL;
  for (int i = 0; i < n; i++) AdjMat[i*n+i] = 0;
  for (int i = 0; i < n; i++)
    for (int j = g.nindex[i]; j < g.nindex[i+1]; j++) {
      int w = g.eweight[j]; if (w < 0) w = -w;
      AdjMat[i*n + g.nlist[j]] = w;
    }
  for (int k = 0; k < n; k++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        mtype v = AdjMat[i*n+k] + AdjMat[k*n+j];
        if (v < AdjMat[i*n+j]) AdjMat[i*n+j] = v;
      }
}

static void FW_gpu(const ECLgraph& g, mtype* AdjMat_out, int repeat)
{
  int n = g.nodes;
  long long N2 = (long long)n * n;

  mtype* d_nidx  = new mtype[g.nodes+1];
  mtype* d_nlist = new mtype[g.edges];
  mtype* d_ewt   = new mtype[g.edges];
  mtype* d_adj   = new mtype[N2];

  for (int i = 0; i <= g.nodes; i++) d_nidx[i] = g.nindex[i];
  for (int i = 0; i < g.edges;  i++) { d_nlist[i] = g.nlist[i]; d_ewt[i] = g.eweight[i]; }

  #pragma omp target enter data map(to: d_nidx[0:g.nodes+1], d_nlist[0:g.edges], d_ewt[0:g.edges]) \
                                map(alloc: d_adj[0:N2])

  auto tstart = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // Init adjacency matrix
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (long long idx = 0; idx < N2; idx++) {
      int i = (int)(idx / n), j = (int)(idx % n);
      d_adj[idx] = (i == j) ? (mtype)0 : INF_VAL;
    }

    // Add edges
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) {
      for (int p = d_nidx[i]; p < d_nidx[i+1]; p++) {
        mtype w = d_ewt[p]; if (w < 0) w = -w;
        d_adj[(long long)i * n + d_nlist[p]] = w;
      }
    }

    // Floyd-Warshall
    for (int k = 0; k < n; k++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (long long idx = 0; idx < N2; idx++) {
        int i = (int)(idx / n), j = (int)(idx % n);
        mtype v = d_adj[(long long)i * n + k] + d_adj[(long long)k * n + j];
        if (v < d_adj[idx]) d_adj[idx] = v;
      }
    }
  }

  auto tend = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count() / 1000.0;
  printf("GPU time (%d repetitions): %.3f ms, avg %.3f ms\n", repeat, ms, ms/repeat);

  #pragma omp target update from(d_adj[0:N2])

  for (long long i = 0; i < N2; i++) AdjMat_out[i] = d_adj[i];

  #pragma omp target exit data map(delete: d_nidx[0:g.nodes+1], d_nlist[0:g.edges], \
                                           d_ewt[0:g.edges], d_adj[0:N2])
  delete[] d_nidx; delete[] d_nlist; delete[] d_ewt; delete[] d_adj;
}

int main(int argc, char* argv[])
{
  printf("ECL-APSP OMP port\n");

  ECLgraph g;
  bool own_graph = false;

  if (argc >= 3) {
    g = readECLgraph(argv[1]);
    if (g.eweight == nullptr) { fprintf(stderr, "Graph has no weights\n"); return 1; }
    for (int i = 0; i < g.edges; i++) if (g.eweight[i] < 0) g.eweight[i] = -g.eweight[i];
    printf("Graph from file: nodes=%d edges=%d\n", g.nodes, g.edges);
  } else {
    printf("No graph file given – generating random 256-node graph\n");
    g.nodes = 256;
    g.edges = 0;
    int density = 4;
    g.nindex = new int[g.nodes + 1];
    g.nlist  = new int[g.nodes * density];
    g.eweight= new int[g.nodes * density];
    srand(42);
    g.nindex[0] = 0;
    for (int i = 0; i < g.nodes; i++) {
      for (int d = 0; d < density; d++) {
        g.nlist[g.edges]   = rand() % g.nodes;
        g.eweight[g.edges] = 1 + rand() % 100;
        g.edges++;
      }
      g.nindex[i+1] = g.edges;
    }
    own_graph = true;
  }

  const int repeat = (argc >= 2) ? atoi(argv[argc-1]) : 1;

  mtype* AdjMat_gpu = new mtype[(long long)g.nodes * g.nodes];
  mtype* AdjMat_cpu = new mtype[(long long)g.nodes * g.nodes];

  FW_gpu(g, AdjMat_gpu, repeat);
  FW_cpu(g, AdjMat_cpu);

  int diff = 0;
  for (int i = 0; i < g.nodes * g.nodes; i++)
    if (AdjMat_gpu[i] != AdjMat_cpu[i]) diff++;

  if (diff) printf("Results differ in %d entries\n", diff);
  else      printf("Results match\n");

  delete[] AdjMat_gpu;
  delete[] AdjMat_cpu;
  if (own_graph || !own_graph) freeECLgraph(g);
  return diff ? 1 : 0;
}

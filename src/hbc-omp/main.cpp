/*
Betweenness Centrality (BC) computation on graphs.
OpenMP target offloading port.

Algorithm (Brandes):
  For each source vertex s:
    1. BFS: compute d[v] = distance, sigma[v] = number of shortest paths
    2. Backward: delta[v] = dependency score
    3. BC[v] += delta[v] / 2
*/

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <climits>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

struct Graph {
    int n, m;
    int *R, *C;
};

static Graph makeRandomGraph(int n, int target_edges, unsigned seed)
{
    srand(seed);
    std::set<std::pair<int,int>> edgeSet;
    while ((int)edgeSet.size() < target_edges) {
        int u = rand() % n, v = rand() % n;
        if (u == v) continue;
        edgeSet.insert({u, v});
        edgeSet.insert({v, u});
    }
    std::vector<std::vector<int>> adj(n);
    for (auto& e : edgeSet) adj[e.first].push_back(e.second);

    Graph g;
    g.n = n;
    g.R = new int[n + 1];
    g.R[0] = 0;
    for (int i = 0; i < n; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        g.R[i+1] = g.R[i] + (int)adj[i].size();
    }
    g.m = g.R[n];
    g.C = new int[g.m];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)adj[i].size(); j++)
            g.C[g.R[i] + j] = adj[i][j];
    return g;
}

static Graph parseGraph(const char* fname)
{
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    std::vector<std::pair<int,int>> edges;
    int maxNode = -1, u, v;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%d %d", &u, &v) == 2 ||
            sscanf(line, "%d,%d", &u, &v) == 2) {
            if (u == v) continue;
            edges.push_back({u, v});
            edges.push_back({v, u});
            maxNode = std::max(maxNode, std::max(u, v));
        }
    }
    fclose(f);
    int n = maxNode + 1;
    std::vector<std::vector<int>> adj(n);
    for (auto& e : edges) adj[e.first].push_back(e.second);

    Graph g;
    g.n = n;
    g.R = new int[n + 1];
    g.R[0] = 0;
    for (int i = 0; i < n; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        g.R[i+1] = g.R[i] + (int)adj[i].size();
    }
    g.m = g.R[n];
    g.C = new int[g.m];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)adj[i].size(); j++)
            g.C[g.R[i] + j] = adj[i][j];
    return g;
}

int main(int argc, char* argv[])
{
    const char* infile = nullptr;
    int  k_src  = -1;
    bool approx = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i+1 < argc)  infile = argv[++i];
        else if (!strcmp(argv[i], "-k") && i+1 < argc) k_src = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--approx"))  approx = true;
    }

    Graph g = infile ? parseGraph(infile) : makeRandomGraph(1000, 5000, 42);
    printf("Number of nodes: %d\n", g.n);
    printf("Number of edges: %d\n", g.m);

    const int n = g.n, m = g.m;

    std::vector<int> sources;
    if (approx && k_src > 0) {
        srand(123);
        std::set<int> chosen;
        while ((int)chosen.size() < std::min(k_src, n)) chosen.insert(rand() % n);
        sources.assign(chosen.begin(), chosen.end());
    } else {
        sources.resize(n);
        for (int i = 0; i < n; i++) sources[i] = i;
        if (k_src > 0 && k_src < n) sources.resize(k_src);
    }
    const int nSrc = (int)sources.size();

    int*       d_R     = g.R;
    int*       d_C     = g.C;
    int*       d_dist  = (int*)malloc(n * sizeof(int));
    long long* d_sigma = (long long*)malloc(n * sizeof(long long));
    double*    d_delta = (double*)malloc(n * sizeof(double));
    float*     d_bc    = (float*)malloc(n * sizeof(float));
    int*       d_Q     = (int*)malloc(n * sizeof(int));
    int*       d_nextQ = (int*)malloc(n * sizeof(int));
    int*       d_Qsz   = (int*)malloc(1 * sizeof(int));
    int*       d_NQsz  = (int*)malloc(1 * sizeof(int));
    int*       d_stack = (int*)malloc(n * sizeof(int));

    #pragma omp target enter data map(to: d_R[0:n+1], d_C[0:m]) \
        map(alloc: d_dist[0:n], d_sigma[0:n], d_delta[0:n], d_bc[0:n], \
                  d_Q[0:n], d_nextQ[0:n], d_Qsz[0:1], d_NQsz[0:1], d_stack[0:n])

    // Initialize bc to 0
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < n; i++) d_bc[i] = 0.0f;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int si = 0; si < nSrc; si++) {
        const int s = sources[si];

        // Initialize per-source arrays
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < n; v++) {
            d_dist[v]  = -1;
            d_sigma[v] = 0LL;
            d_delta[v] = 0.0;
        }

        // Initialize source
        #pragma omp target
        {
            d_dist[s]  = 0;
            d_sigma[s] = 1LL;
            d_Q[0]     = s;
            d_Qsz[0]   = 1;
        }

        std::vector<int> levelBeg;
        levelBeg.push_back(0);

        int stackPtr = 0;
        int qsize    = 1;

        // BFS forward pass
        while (qsize > 0) {
            const int sp = stackPtr;
            const int qs = qsize;

            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int i = 0; i < qs; i++) d_stack[sp + i] = d_Q[i];

            stackPtr += qsize;
            levelBeg.push_back(stackPtr);

            #pragma omp target
            { d_NQsz[0] = 0; }

            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int fi = 0; fi < qs; fi++) {
                const int u  = d_Q[fi];
                const int du = d_dist[u];
                for (int e = d_R[u]; e < d_R[u+1]; e++) {
                    const int v = d_C[e];
                    int expected = -1;
                    int got;
                    #pragma omp atomic capture
                    { got = d_dist[v]; if (d_dist[v] == -1) d_dist[v] = du + 1; }
                    if (got == -1) {
                        int idx;
                        #pragma omp atomic capture
                        idx = d_NQsz[0]++;
                        d_nextQ[idx] = v;
                    }
                    if (d_dist[v] == du + 1) {
                        #pragma omp atomic update
                        d_sigma[v] += d_sigma[u];
                    }
                }
            }

            // Swap frontiers
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int i = 0; i < n; i++) {
                if (i < d_NQsz[0]) d_Q[i] = d_nextQ[i];
            }

            #pragma omp target update from(d_NQsz[0:1])
            qsize = d_NQsz[0];
        }

        const int numLevels = (int)levelBeg.size() - 1;

        // BFS backward pass
        for (int L = numLevels - 1; L >= 0; L--) {
            const int beg = levelBeg[L];
            const int end = levelBeg[L + 1];
            const int len = end - beg;
            if (len <= 0) continue;

            const int base = beg;
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int ii = 0; ii < len; ii++) {
                const int u = d_stack[base + ii];
                double dep = 0.0;
                for (int e = d_R[u]; e < d_R[u+1]; e++) {
                    const int v = d_C[e];
                    if (d_dist[v] == d_dist[u] + 1 && d_sigma[v] > 0LL) {
                        dep += ((double)d_sigma[u] / (double)d_sigma[v])
                               * (1.0 + d_delta[v]);
                    }
                }
                d_delta[u] = dep;
            }
        }

        // Accumulate BC (exclude source)
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < n; v++) {
            if (v != s) d_bc[v] += (float)(d_delta[v] / 2.0);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    float GPU_time_us =
        (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("Time for GPU execution: %.9f s\n", GPU_time_us / 1.0e6f);

    #pragma omp target exit data map(delete: d_R[0:n+1], d_C[0:m], d_dist[0:n], \
        d_sigma[0:n], d_delta[0:n], d_bc[0:n], d_Q[0:n], d_nextQ[0:n], \
        d_Qsz[0:1], d_NQsz[0:1], d_stack[0:n])

    free(d_dist); free(d_sigma); free(d_delta); free(d_bc);
    free(d_Q); free(d_nextQ); free(d_Qsz); free(d_NQsz); free(d_stack);
    delete[] g.R;
    delete[] g.C;
    return 0;
}

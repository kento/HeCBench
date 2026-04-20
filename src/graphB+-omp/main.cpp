/*
graphB+ balancing algorithm - OpenMP target offloading port
*/

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <climits>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <tuple>

static void readGraph(const char* filename,
                      int& N, int& E,
                      std::vector<int>& nindex,
                      std::vector<int>& nlist,
                      std::vector<int>& eweight,
                      std::vector<int>& origID)
{
    FILE* fin = fopen(filename, "rt");
    if (!fin) { fprintf(stderr, "ERROR: cannot open %s\n", filename); exit(1); }

    char buf[256];
    fgets(buf, sizeof(buf), fin);

    std::map<int,int> nodeMap;
    std::set<std::pair<int,int>> seen;
    std::vector<std::tuple<int,int,int>> edgeList;
    int cnt = 0;

    int src, dst, wei;
    while (fscanf(fin, "%d,%d,%d", &src, &dst, &wei) == 3) {
        if (src == dst) continue;
        if (wei < -1 || wei > 1) continue;
        int a = std::min(src, dst), b = std::max(src, dst);
        if (seen.count({a, b})) continue;
        seen.insert({a, b});
        if (!nodeMap.count(src)) nodeMap[src] = cnt++;
        if (!nodeMap.count(dst)) nodeMap[dst] = cnt++;
        edgeList.push_back({nodeMap[src], nodeMap[dst], wei});
    }
    fclose(fin);

    N = cnt;
    E = (int)edgeList.size() * 2;

    std::vector<int> deg(N, 0);
    for (auto& [u, v, w] : edgeList) { deg[u]++; deg[v]++; }

    nindex.resize(N + 1);
    nindex[0] = 0;
    for (int i = 0; i < N; i++) nindex[i+1] = nindex[i] + deg[i];

    nlist.resize(E);
    eweight.resize(E);
    std::vector<int> pos(N, 0);
    for (auto& [u, v, w] : edgeList) {
        nlist  [nindex[u] + pos[u]] = v;
        eweight[nindex[u] + pos[u]] = w;
        pos[u]++;
        nlist  [nindex[v] + pos[v]] = u;
        eweight[nindex[v] + pos[v]] = w;
        pos[v]++;
    }

    origID.resize(N);
    for (auto& [orig, mapped] : nodeMap) origID[mapped] = orig;
}

int main(int argc, char* argv[])
{
    printf("graphB+ balancing code for signed social network graphs\n");

    if (argc != 4) {
        printf("USAGE: %s input_file iterations output_file\n", argv[0]);
        return 1;
    }

    printf("verification is off\n");

    int N = 0, E = 0;
    std::vector<int> h_nindex, h_nlist, h_eweight, h_origID;
    readGraph(argv[1], N, E, h_nindex, h_nlist, h_eweight, h_origID);

    printf("nodes: %d\n", N);
    printf("edges: %d\n", E);

    const int iterations = atoi(argv[2]);

    std::vector<int> root(N);
    for (int i = 0; i < N; i++) root[i] = i;
    std::partial_sort(root.begin(),
                      root.begin() + std::min(iterations, N),
                      root.end(),
                      [&](int a, int b){
                          return (h_nindex[a+1]-h_nindex[a]) > (h_nindex[b+1]-h_nindex[b]);
                      });

    int* d_nindex  = (int*)malloc((N + 1) * sizeof(int));
    int* d_nlist   = (int*)malloc(E * sizeof(int));
    int* d_eweight = (int*)malloc(E * sizeof(int));
    int* d_parent  = (int*)malloc(N * sizeof(int));
    int* d_color   = (int*)malloc(N * sizeof(int));
    int* d_frontier= (int*)malloc(N * sizeof(int));
    int* d_nextFr  = (int*)malloc(N * sizeof(int));
    int* d_fsize   = (int*)malloc(1 * sizeof(int));
    int* d_nsize   = (int*)malloc(1 * sizeof(int));
    int* d_label   = (int*)malloc(N * sizeof(int));
    int* d_ccCnt   = (int*)malloc(N * sizeof(int));
    int* d_hop     = (int*)malloc(N * sizeof(int));
    int* d_inCC    = (int*)malloc(N * sizeof(int));
    int* d_inTree  = (int*)malloc(E * sizeof(int));
    int* d_negCnt  = (int*)malloc(E * sizeof(int));

    for (int i = 0; i <= N; i++) d_nindex[i] = h_nindex[i];
    for (int i = 0; i < E;  i++) d_nlist[i]  = h_nlist[i];
    for (int i = 0; i < E;  i++) d_eweight[i]= h_eweight[i];
    for (int i = 0; i < N;  i++) { d_inCC[i] = 0; }
    for (int i = 0; i < E;  i++) { d_inTree[i] = 0; d_negCnt[i] = 0; }

    #pragma omp target enter data \
      map(to: d_nindex[0:N+1], d_nlist[0:E], d_eweight[0:E]) \
      map(alloc: d_parent[0:N], d_color[0:N], d_frontier[0:N], d_nextFr[0:N], \
                 d_fsize[0:1], d_nsize[0:1], d_label[0:N], d_ccCnt[0:N], d_hop[0:N]) \
      map(tofrom: d_inCC[0:N], d_inTree[0:E], d_negCnt[0:E])

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        const int r = root[iter % N];

        // Step 1: BFS spanning tree + 2-coloring
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < N; i++) {
            d_parent[i] = -1;
            d_color [i] = -1;
        }

        // Init root on device
        #pragma omp target
        {
            d_parent[r]   = r;
            d_color [r]   = 0;
            d_frontier[0] = r;
            d_fsize[0]    = 1;
            d_nsize[0]    = 0;
        }

        int fsize = 1;
        while (fsize > 0) {
            #pragma omp target
            { d_nsize[0] = 0; }

            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int fi = 0; fi < fsize; fi++) {
                const int node      = d_frontier[fi];
                const int nodeColor = d_color[node];
                for (int j = d_nindex[node]; j < d_nindex[node+1]; j++) {
                    const int nbr = d_nlist[j];
                    int old;
                    #pragma omp atomic compare capture
                    { old = d_parent[nbr]; if (d_parent[nbr] == -1) d_parent[nbr] = node; }
                    if (old == -1) {
                        int idx;
                        #pragma omp atomic capture
                        { idx = d_nsize[0]; d_nsize[0]++; }
                        d_nextFr[idx] = nbr;
                        d_color[nbr] = (d_eweight[j] < 0) ? (1 - nodeColor) : nodeColor;
                    }
                }
            }

            // Swap frontiers
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int i = 0; i < N; i++) {
                if (i < d_nsize[0]) d_frontier[i] = d_nextFr[i];
            }

            #pragma omp target update from(d_nsize[0:1])
            fsize = d_nsize[0];
        }

        // Step 2: Update inTree and negCnt
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < N; v++) {
            const int par = d_parent[v];
            if (par < 0 || par == v) continue;
            for (int j = d_nindex[v]; j < d_nindex[v+1]; j++) {
                if (d_nlist[j] == par) {
                    #pragma omp atomic update
                    d_inTree[j] += 1;
                    const int w  = d_eweight[j];
                    const int cv = d_color[v], cp = d_color[par];
                    bool ok = (w >= 0) ? (cv == cp) : (cv != cp);
                    if (!ok) {
                        #pragma omp atomic update
                        d_negCnt[j] += 1;
                    }
                    break;
                }
            }
        }

        // Step 3: Union-find on balanced edges
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < N; v++) d_label[v] = v;

        for (int round = 0; round < 20; round++) {
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int v = 0; v < N; v++) {
                for (int j = d_nindex[v]; j < d_nindex[v+1]; j++) {
                    const int u = d_nlist[j];
                    const int w = d_eweight[j];
                    const int cv = d_color[v], cu = d_color[u];
                    bool balanced = (w >= 0) ? (cv == cu) : (cv != cu);
                    if (!balanced) continue;
                    int lv = d_label[v], lu = d_label[u];
                    while (lv != lu) {
                        if (lv > lu) {
                            int old;
                            #pragma omp atomic compare capture
                            { old = d_label[lv]; if (d_label[lv] == lv) d_label[lv] = lu; }
                            if (old == lv) break;
                            lv = old;
                        } else {
                            int old;
                            #pragma omp atomic compare capture
                            { old = d_label[lu]; if (d_label[lu] == lu) d_label[lu] = lv; }
                            if (old == lu) break;
                            lu = old;
                        }
                    }
                }
            }

            // Path compression
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int v = 0; v < N; v++) {
                int root_v = d_label[v];
                while (root_v != d_label[root_v]) root_v = d_label[root_v];
                d_label[v] = root_v;
            }
        }

        // Step 4: Count CC sizes
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < N; v++) d_ccCnt[v] = 0;

        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < N; v++) {
            #pragma omp atomic update
            d_ccCnt[d_label[v]] += 1;
        }

        int max_cc_size = 0;
        #pragma omp target teams distribute parallel for reduction(max:max_cc_size) thread_limit(256)
        for (int v = 0; v < N; v++) {
            if (d_ccCnt[v] > max_cc_size) max_cc_size = d_ccCnt[v];
        }

        const int mcs = max_cc_size;
        int largest_lbl = N - 1;
        #pragma omp target teams distribute parallel for reduction(min:largest_lbl) thread_limit(256)
        for (int v = 0; v < N; v++) {
            if (d_ccCnt[v] == mcs && v < largest_lbl) largest_lbl = v;
        }

        // Step 5: BFS/Bellman-Ford hop distances
        const int INF_HOP = INT_MAX - 1;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < N; v++) d_hop[v] = INF_HOP;

        const int ll = largest_lbl;
        #pragma omp target
        { d_hop[ll] = 0; }

        for (int bfIter = 0; bfIter < 30; bfIter++) {
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int v = 0; v < N; v++) {
                const int lv = d_label[v];
                for (int j = d_nindex[v]; j < d_nindex[v+1]; j++) {
                    const int u  = d_nlist[j];
                    const int lu = d_label[u];
                    if (lv == lu) continue;
                    if (d_hop[lv] != INF_HOP) {
                        int newDist = d_hop[lv] + 1;
                        #pragma omp atomic compare
                        if (d_hop[lu] > newDist) d_hop[lu] = newDist;
                    }
                    if (d_hop[lu] != INF_HOP) {
                        int newDist = d_hop[lu] + 1;
                        #pragma omp atomic compare
                        if (d_hop[lv] > newDist) d_hop[lv] = newDist;
                    }
                }
            }
        }

        // Step 6: Increment inCC for nodes at even hop distance
        const int INF2 = INF_HOP;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int v = 0; v < N; v++) {
            const int h = d_hop[d_label[v]];
            if (h != INF2 && (h % 2) == 0) {
                #pragma omp atomic update
                d_inCC[v] += 1;
            }
        }
    } // end iterations

    auto t_end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t_end - t_start).count();
    printf("Runtime: %.3f s\n", runtime);

    #pragma omp target update from(d_inCC[0:N], d_inTree[0:E], d_negCnt[0:E])

    FILE* fout = fopen(argv[3], "wt");
    if (!fout) { fprintf(stderr, "ERROR: cannot open output file %s\n", argv[3]); }
    else {
        fprintf(fout, "original node ID, percentage node was in agreeable majority\n");
        for (int i = 0; i < N; i++)
            fprintf(fout, "%d,%.1f\n", h_origID[i], 100.0 * d_inCC[i] / iterations);

        fprintf(fout, "source node ID, destination node ID, percentage edge was in tree, percentage edge was negative\n");
        for (int v = 0; v < N; v++) {
            for (int j = h_nindex[v]; j < h_nindex[v+1]; j++) {
                const int u = h_nlist[j];
                if (v < u) {
                    fprintf(fout, "%d,%d,%.1f,%.1f\n",
                        h_origID[v], h_origID[u],
                        100.0 * d_inTree[j] / iterations,
                        100.0 * d_negCnt[j] / iterations);
                }
            }
        }
        fclose(fout);
    }

    #pragma omp target exit data \
      map(delete: d_nindex[0:N+1], d_nlist[0:E], d_eweight[0:E], \
                  d_parent[0:N], d_color[0:N], d_frontier[0:N], d_nextFr[0:N], \
                  d_fsize[0:1], d_nsize[0:1], d_label[0:N], d_ccCnt[0:N], d_hop[0:N], \
                  d_inCC[0:N], d_inTree[0:E], d_negCnt[0:E])

    free(d_nindex); free(d_nlist); free(d_eweight);
    free(d_parent); free(d_color); free(d_frontier); free(d_nextFr);
    free(d_fsize); free(d_nsize); free(d_label); free(d_ccCnt); free(d_hop);
    free(d_inCC); free(d_inTree); free(d_negCnt);
    return 0;
}

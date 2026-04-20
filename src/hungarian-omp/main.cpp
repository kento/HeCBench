#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <chrono>
#include <vector>
#include <random>

static const int user_n  = 1000;
static const int n       = 1024;  // nextPow2(user_n)
static const int range   = n;
static const int n_tests = 10;

// Subtract row minimums using OMP target
static void subtractRowMin(int* cost, int N) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int r = 0; r < N; r++) {
        int mn = INT_MAX;
        for (int c = 0; c < N; c++) {
            int v = cost[r * N + c];
            if (v < mn) mn = v;
        }
        for (int c = 0; c < N; c++)
            cost[r * N + c] -= mn;
    }
}

// Subtract column minimums using OMP target
static void subtractColMin(int* cost, int N) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int c = 0; c < N; c++) {
        int mn = INT_MAX;
        for (int r = 0; r < N; r++) {
            int v = cost[r * N + c];
            if (v < mn) mn = v;
        }
        for (int r = 0; r < N; r++)
            cost[r * N + c] -= mn;
    }
}

// Sequential Hungarian on host
static void hungarianCPU(const std::vector<int>& costIn, int N, std::vector<int>& assign) {
    std::vector<int> cost(costIn);

    assign.assign(N, -1);
    std::vector<int> rowAssign(N, -1);

    std::vector<bool> usedRow(N, false), usedCol(N, false);
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            if (cost[r * N + c] == 0 && !usedRow[r] && !usedCol[c]) {
                assign[r]    = c;
                rowAssign[c] = r;
                usedRow[r]   = true;
                usedCol[c]   = true;
                break;
            }
        }
    }

    for (int r0 = 0; r0 < N; r0++) {
        if (assign[r0] >= 0) continue;

        std::vector<int>  dist(N, INT_MAX);
        std::vector<int>  prevCol(N, -1);
        std::vector<int>  prevRow(N, -1);
        std::vector<bool> visited(N, false);

        for (int c = 0; c < N; c++)
            dist[c] = cost[r0 * N + c];

        int endCol = -1;
        while (true) {
            int minD = INT_MAX, j = -1;
            for (int c = 0; c < N; c++) {
                if (!visited[c] && dist[c] < minD) { minD = dist[c]; j = c; }
            }
            if (j < 0) break;
            visited[j] = true;
            if (rowAssign[j] < 0) { endCol = j; break; }

            int r1 = rowAssign[j];
            for (int c = 0; c < N; c++) {
                if (!visited[c]) {
                    int nd = minD + cost[r1 * N + c] - cost[r1 * N + j];
                    if (nd < 0) nd = 0;
                    if (nd < dist[c]) {
                        dist[c]   = nd;
                        prevCol[c] = j;
                        prevRow[c] = r1;
                    }
                }
            }
        }

        if (endCol < 0) continue;

        int c = endCol;
        while (c >= 0) {
            int pc = prevCol[c];
            int pr = (pc >= 0) ? prevRow[c] : r0;
            assign[pr]    = c;
            rowAssign[c]  = pr;
            c = pc;
        }
    }
}

int main(int /*argc*/, char ** /*argv*/) {
    printf("Starting Hungarian algorithm benchmarks\n");
    printf("n=%d, n_tests=%d\n", n, n_tests);

    std::mt19937 rng(45345);
    std::uniform_int_distribution<int> dist(0, range - 1);

    std::vector<int> h_cost(n * n);
    for (int i = 0; i < n * n; i++)
        h_cost[i] = dist(rng);

    double totalMs = 0.0;

    int* d_cost = (int*)malloc(n * n * sizeof(int));
    #pragma omp target enter data map(alloc: d_cost[0:n*n])

    for (int t = 0; t < n_tests; t++) {
        // Fresh copy of cost matrix
        memcpy(d_cost, h_cost.data(), n * n * sizeof(int));
        #pragma omp target update to(d_cost[0:n*n])

        auto t0 = std::chrono::high_resolution_clock::now();

        subtractRowMin(d_cost, n);
        subtractColMin(d_cost, n);

        // Copy reduced cost back to host
        #pragma omp target update from(d_cost[0:n*n])
        std::vector<int> h_reduced(d_cost, d_cost + n*n);

        // Sequential Hungarian on host
        std::vector<int> assign;
        hungarianCPU(h_reduced, n, assign);

        auto t1 = std::chrono::high_resolution_clock::now();
        totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

        int matches = 0;
        std::vector<bool> usedCol(n, false);
        for (int r = 0; r < n; r++) {
            int c = assign[r];
            if (c >= 0 && c < n && !usedCol[c]) {
                usedCol[c] = true;
                matches++;
            }
        }
        if (matches != n) {
            printf("Test %d: WARNING — only %d matches (expected %d)\n",
                   t, matches, n);
        }
    }

    printf("Average time: %.3f ms\n", totalMs / n_tests);

    #pragma omp target exit data map(delete: d_cost[0:n*n])
    free(d_cost);
    return 0;
}

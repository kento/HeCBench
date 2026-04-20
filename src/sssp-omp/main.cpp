// OpenMP target offloading port of sssp benchmark
// Single-Source Shortest Path using Bellman-Ford on GPU
// Reads graph from file (same format as original benchmark)

#include <omp.h>
#include <unistd.h>
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <fstream>
#include <sstream>
#include <atomic>

#define INF (-2147483647)
#define UP_LIMIT 16677216
#define WHITE 16677217
#define GRAY  16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221
#define W_QUEUE_SIZE 1600

typedef struct { int x; int y; } Node;
typedef struct { int x; int y; } Edge;

struct Params {
  int n_gpu_threads;
  int n_gpu_blocks;
  int n_threads;
  int n_warmup;
  int n_reps;
  const char *file_name;
  const char *comparison_file;
  int switching_limit;

  Params(int argc, char **argv) {
    n_gpu_threads   = 256;
    n_gpu_blocks    = 8;
    n_threads       = 2;
    n_warmup        = 1;
    n_reps          = 1;
    file_name       = "input/NYR_input.dat";
    comparison_file = "output/NYR_bfs.out";
    switching_limit = 128;
    int opt;
    while ((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:")) >= 0) {
      switch (opt) {
        case 'i': n_gpu_threads   = atoi(optarg); break;
        case 'g': n_gpu_blocks    = atoi(optarg); break;
        case 't': n_threads       = atoi(optarg); break;
        case 'w': n_warmup        = atoi(optarg); break;
        case 'r': n_reps          = atoi(optarg); break;
        case 'f': file_name       = optarg; break;
        case 'c': comparison_file = optarg; break;
        case 'l': switching_limit = atoi(optarg); break;
        default: break;
      }
    }
  }
};

static bool read_graph(const char *fname, int &n_nodes, int &n_edges,
                       std::vector<Node> &nodes, std::vector<Edge> &edges,
                       std::vector<int> &cost, std::vector<int> &color) {
  FILE *fp = fopen(fname, "r");
  if (!fp) { printf("Error opening file %s\n", fname); return false; }
  fscanf(fp, "%d %d", &n_nodes, &n_edges);
  nodes.resize(n_nodes);
  edges.resize(n_edges);
  cost.resize(n_nodes, INF);
  color.resize(n_nodes, WHITE);
  for (int i = 0; i < n_nodes; i++) {
    fscanf(fp, "%d %d", &nodes[i].x, &nodes[i].y);
  }
  for (int i = 0; i < n_edges; i++) {
    fscanf(fp, "%d %d", &edges[i].x, &edges[i].y);
  }
  fclose(fp);
  return true;
}

static bool verify(const int *h_cost, int n_nodes, const char *file_name) {
  FILE *fpo = fopen(file_name, "r");
  if (!fpo) { printf("Error Reading output file\n"); return false; }
  int num_nodes_o = 0;
  fscanf(fpo, "%d", &num_nodes_o);
  if (n_nodes != num_nodes_o) { printf("FAIL: node count mismatch\n"); fclose(fpo); return false; }
  for (int i = 0; i < n_nodes; i++) {
    int expected;
    fscanf(fpo, "%d", &expected);
    if (h_cost[i] != expected) {
      printf("FAIL at node %d: got %d expected %d\n", i, h_cost[i], expected);
      fclose(fpo); return false;
    }
  }
  fclose(fpo);
  return true;
}

// GPU SSSP kernel (one iteration of Bellman-Ford / BFS wavefront)
static void sssp_kernel(Node *nodes, Edge *edges, int *cost, int *color,
                        int *q_in, int *q_out, int *num_in,
                        int *head, int *tail, int *overflow,
                        int *gray_shade, int *iter,
                        int n_gpu_threads, int n_gpu_blocks) {
  int total_threads = n_gpu_blocks * n_gpu_threads;
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int tid = 0; tid < total_threads; tid++) {
    int iter_val = *iter;
    int gshade   = *gray_shade;
    int next_gshade = (gshade == GRAY0) ? GRAY1 : GRAY0;

    // Process queue entries assigned to this thread
    while (true) {
      int my_pos;
      #pragma omp atomic capture
      { my_pos = *head; (*head)++; }
      if (my_pos >= *num_in) break;

      int node_id = q_in[my_pos];
      if (color[node_id] != gshade) continue;

      color[node_id] = BLACK;
      int start = nodes[node_id].x;
      int num_edges = nodes[node_id].y;

      for (int e = start; e < start + num_edges; e++) {
        int neighbor = edges[e].x;
        int weight   = edges[e].y;
        int new_cost = cost[node_id] + weight;
        if (new_cost < cost[neighbor]) {
          #pragma omp atomic write
          cost[neighbor] = new_cost;
          if (color[neighbor] != next_gshade) {
            color[neighbor] = next_gshade;
            int out_pos;
            #pragma omp atomic capture
            { out_pos = *tail; (*tail)++; }
            if (out_pos < W_QUEUE_SIZE * n_gpu_blocks) {
              q_out[out_pos] = neighbor;
            } else {
              #pragma omp atomic write
              *overflow = 1;
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  Params p(argc, argv);

  int n_nodes = 0, n_edges = 0;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<int>  h_cost, h_color;

  if (!read_graph(p.file_name, n_nodes, n_edges, nodes, edges, h_cost, h_color)) {
    printf("Failed to read graph\n");
    return 1;
  }

  // Source node
  h_cost[0]  = 0;
  h_color[0] = GRAY0;

  std::vector<int> h_q1(n_nodes, 0), h_q2(n_nodes, 0);
  h_q1[0] = 0;

  Node *d_nodes = nodes.data();
  Edge *d_edges = edges.data();
  int  *d_cost  = h_cost.data();
  int  *d_color = h_color.data();
  int  *d_q1    = h_q1.data();
  int  *d_q2    = h_q2.data();

  int h_num_t[1] = {1};
  int h_head[1]  = {0};
  int h_tail[1]  = {0};
  int h_overflow[1] = {0};
  int h_iter[1]     = {0};
  int h_gray[1]     = {GRAY0};
  int *d_num_t   = h_num_t;
  int *d_head    = h_head;
  int *d_tail    = h_tail;
  int *d_overflow= h_overflow;
  int *d_iter    = h_iter;
  int *d_gray    = h_gray;

  #pragma omp target enter data \
    map(to:   d_nodes[0:n_nodes], d_edges[0:n_edges]) \
    map(tofrom: d_cost[0:n_nodes], d_color[0:n_nodes]) \
    map(tofrom: d_q1[0:n_nodes],   d_q2[0:n_nodes]) \
    map(tofrom: d_num_t[0:1], d_head[0:1], d_tail[0:1], \
                d_overflow[0:1], d_iter[0:1], d_gray[0:1])

  auto t_start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
    // Reset
    if (rep > 0) {
      for (int i = 0; i < n_nodes; i++) { h_cost[i] = INF; h_color[i] = WHITE; }
      h_cost[0] = 0; h_color[0] = GRAY0;
      for (int i = 0; i < n_nodes; i++) { h_q1[i] = 0; h_q2[i] = 0; }
      h_q1[0] = 0; h_num_t[0] = 1; h_iter[0] = 0; h_gray[0] = GRAY0;
      #pragma omp target update to(d_cost[0:n_nodes], d_color[0:n_nodes], \
                                   d_q1[0:n_nodes],   d_q2[0:n_nodes], \
                                   d_num_t[0:1], d_iter[0:1], d_gray[0:1])
    }

    // BFS wavefront iterations
    while (h_num_t[0] > 0) {
      h_head[0] = 0; h_tail[0] = 0; h_overflow[0] = 0;
      #pragma omp target update to(d_head[0:1], d_tail[0:1], d_overflow[0:1])

      int *q_in  = (h_iter[0] % 2 == 0) ? d_q1 : d_q2;
      int *q_out = (h_iter[0] % 2 == 0) ? d_q2 : d_q1;

      sssp_kernel(d_nodes, d_edges, d_cost, d_color, q_in, q_out,
                  d_num_t, d_head, d_tail, d_overflow, d_gray, d_iter,
                  p.n_gpu_threads, p.n_gpu_blocks);

      #pragma omp target update from(d_tail[0:1], d_iter[0:1])
      h_num_t[0] = h_tail[0];
      h_iter[0]++;
      h_gray[0] = (h_iter[0] % 2 == 0) ? GRAY0 : GRAY1;
      #pragma omp target update to(d_num_t[0:1], d_iter[0:1], d_gray[0:1])
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
  printf("Total execution time: %f (s)\n", elapsed);

  #pragma omp target update from(d_cost[0:n_nodes], d_color[0:n_nodes])
  #pragma omp target exit data \
    map(delete: d_nodes[0:n_nodes], d_edges[0:n_edges], d_cost[0:n_nodes], d_color[0:n_nodes], \
                d_q1[0:n_nodes], d_q2[0:n_nodes], \
                d_num_t[0:1], d_head[0:1], d_tail[0:1], \
                d_overflow[0:1], d_iter[0:1], d_gray[0:1])

  bool ok = verify(h_cost.data(), n_nodes, p.comparison_file);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}

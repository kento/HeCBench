// OpenMP target offloading port of voxelization benchmark
// LiDAR point cloud voxelization with hash table

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

static const int FEATURE_NUM = 5;
static const float MIN_X = -54.f, MAX_X = 54.f;
static const float MIN_Y = -54.f, MAX_Y = 54.f;
static const float MIN_Z = -5.0f, MAX_Z = 3.0f;
static const float PIL_X = 0.075f, PIL_Y = 0.075f, PIL_Z = 0.2f;
static const int   MAX_VOXELS = 160000;
static const int   MAX_PPV    = 10;

#pragma omp declare target
static uint64_t hash_fn(uint64_t k) {
  k ^= k >> 16; k *= 0x85ebca6b; k ^= k >> 13; k *= 0xc2b2ae35; k ^= k >> 16;
  return k;
}

static void insertHash(uint32_t key, uint32_t *ht, uint32_t ht_size, uint32_t *voxel_count) {
  uint64_t hv   = hash_fn(key);
  uint32_t slot = hv % (ht_size / 2);
  uint32_t empty_key = 0xFFFFFFFFu;
  while (true) {
    uint32_t pre;
    bool swapped = false;
    #pragma omp atomic capture
    { pre = ht[slot]; if (pre == empty_key) { ht[slot] = key; swapped = true; } }
    if (swapped) {
      uint32_t idx;
      #pragma omp atomic capture
      { idx = *voxel_count; (*voxel_count)++; }
      ht[slot + ht_size / 2] = idx;
      break;
    } else if (pre == key) {
      break;
    }
    slot = (slot + 1) % (ht_size / 2);
  }
}

static uint32_t lookupHash(uint32_t key, const uint32_t *ht, uint32_t ht_size) {
  uint64_t hv   = hash_fn(key);
  uint32_t slot = hv % (ht_size / 2);
  while (true) {
    if (ht[slot] == key) return ht[slot + ht_size / 2];
    if (ht[slot] == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    slot = (slot + 1) % (ht_size / 2);
  }
}
#pragma omp end declare target

static int getGridX() { return (int)std::round((MAX_X - MIN_X) / PIL_X); }
static int getGridY() { return (int)std::round((MAX_Y - MIN_Y) / PIL_Y); }
static int getGridZ() { return (int)std::round((MAX_Z - MIN_Z) / PIL_Z); }

static void generateVoxels(const float *d_points, int points_size, int repeat) {
  int gx = getGridX(), gy = getGridY(), gz = getGridZ();
  uint32_t ht_size = (uint32_t)points_size * 2 * 2;

  std::vector<uint32_t> h_ht(ht_size);
  std::vector<float>    h_vox_tmp((size_t)MAX_VOXELS * MAX_PPV * FEATURE_NUM, 0.f);
  std::vector<uint32_t> h_vox_num(MAX_VOXELS, 0);
  std::vector<uint32_t> h_real_num(1, 0);
  std::vector<float>    h_features((size_t)MAX_VOXELS * FEATURE_NUM, 0.f);

  uint32_t *d_ht       = h_ht.data();
  float    *d_vox_tmp  = h_vox_tmp.data();
  uint32_t *d_vox_num  = h_vox_num.data();
  uint32_t *d_real_num = h_real_num.data();
  float    *d_features = h_features.data();

  #pragma omp target enter data \
    map(to: d_points[0:points_size*FEATURE_NUM]) \
    map(alloc: d_ht[0:ht_size], d_vox_tmp[0:MAX_VOXELS*MAX_PPV*FEATURE_NUM], \
               d_vox_num[0:MAX_VOXELS], d_real_num[0:1], d_features[0:MAX_VOXELS*FEATURE_NUM])

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // Reset
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (uint32_t i = 0; i < ht_size; i++) d_ht[i] = 0xFFFFFFFFu;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < MAX_VOXELS; i++) d_vox_num[i] = 0;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < 1; i++) d_real_num[i] = 0;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < MAX_VOXELS * MAX_PPV * FEATURE_NUM; i++) d_vox_tmp[i] = 0.f;

    // Build hash table
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < points_size; idx++) {
      float px = d_points[FEATURE_NUM * idx];
      float py = d_points[FEATURE_NUM * idx + 1];
      float pz = d_points[FEATURE_NUM * idx + 2];
      int vx = (int)floorf((px - MIN_X) / PIL_X);
      int vy = (int)floorf((py - MIN_Y) / PIL_Y);
      int vz = (int)floorf((pz - MIN_Z) / PIL_Z);
      if (vx < 0 || vx >= gx || vy < 0 || vy >= gy || vz < 0 || vz >= gz) continue;
      uint32_t voff = (uint32_t)vz * gy * gx + (uint32_t)vy * gx + vx;
      insertHash(voff, d_ht, ht_size, d_real_num);
    }

    // Scatter points to voxels
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < points_size; idx++) {
      float px = d_points[FEATURE_NUM * idx];
      float py = d_points[FEATURE_NUM * idx + 1];
      float pz = d_points[FEATURE_NUM * idx + 2];
      if (px < MIN_X || px >= MAX_X || py < MIN_Y || py >= MAX_Y ||
          pz < MIN_Z || pz >= MAX_Z) continue;
      int vx = (int)floorf((px - MIN_X) / PIL_X);
      int vy = (int)floorf((py - MIN_Y) / PIL_Y);
      int vz = (int)floorf((pz - MIN_Z) / PIL_Z);
      if (vx < 0 || vx >= gx || vy < 0 || vy >= gy || vz < 0 || vz >= gz) continue;
      uint32_t voff = (uint32_t)vz * gy * gx + (uint32_t)vy * gx + vx;
      uint32_t vid  = lookupHash(voff, d_ht, ht_size);
      if (vid >= (uint32_t)MAX_VOXELS) continue;
      uint32_t cur;
      #pragma omp atomic capture
      { cur = d_vox_num[vid]; d_vox_num[vid]++; }
      if (cur < (uint32_t)MAX_PPV) {
        uint32_t dst = vid * (FEATURE_NUM * MAX_PPV) + cur * FEATURE_NUM;
        uint32_t src = idx * FEATURE_NUM;
        for (int f = 0; f < FEATURE_NUM; f++) d_vox_tmp[dst + f] = d_points[src + f];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the voxelization kernel: %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target update from(d_real_num[0:1])
  uint32_t real_voxels = h_real_num[0];
  std::cout << "valid_num: " << real_voxels << std::endl;

  // Feature extraction: average per voxel
  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int vid = 0; vid < (int)real_voxels; vid++) {
      uint32_t valid = d_vox_num[vid];
      if (valid > (uint32_t)MAX_PPV) valid = (uint32_t)MAX_PPV;
      int base = vid * MAX_PPV * FEATURE_NUM;
      for (int f = 0; f < FEATURE_NUM; f++) {
        float s = d_vox_tmp[base + f];
        for (int p = 1; p < (int)valid; p++) s += d_vox_tmp[base + p * FEATURE_NUM + f];
        d_features[vid * FEATURE_NUM + f] = (valid > 0) ? s / valid : 0.f;
      }
    }
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the feature extraction kernel: %f (us)\n", (time * 1e-3f) / repeat);

  #pragma omp target exit data \
    map(delete: d_points[0:points_size*FEATURE_NUM], \
                d_ht[0:ht_size], d_vox_tmp[0:MAX_VOXELS*MAX_PPV*FEATURE_NUM], \
                d_vox_num[0:MAX_VOXELS], d_real_num[0:1], d_features[0:MAX_VOXELS*FEATURE_NUM])
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <repeat> [points_count]\n", argv[0]);
    return 1;
  }
  const int repeat  = atoi(argv[1]);
  const int npoints = (argc >= 3) ? atoi(argv[2]) : 100000;

  std::vector<float> h_points((size_t)npoints * FEATURE_NUM);
  {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dx(MIN_X, MAX_X);
    std::uniform_real_distribution<float> dy(MIN_Y, MAX_Y);
    std::uniform_real_distribution<float> dz(MIN_Z, MAX_Z);
    std::uniform_real_distribution<float> di(0.f, 1.f);
    for (int i = 0; i < npoints; i++) {
      h_points[i * FEATURE_NUM + 0] = dx(rng);
      h_points[i * FEATURE_NUM + 1] = dy(rng);
      h_points[i * FEATURE_NUM + 2] = dz(rng);
      h_points[i * FEATURE_NUM + 3] = di(rng);
      h_points[i * FEATURE_NUM + 4] = di(rng);
    }
  }

  std::cout << "find points num: " << npoints << std::endl;
  const float *d_points = h_points.data();

  // enter data done inside generateVoxels, so just call it
  generateVoxels(d_points, npoints, repeat);
  return 0;
}

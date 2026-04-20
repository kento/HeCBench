// ssim – OpenMP target port of ssim-cuda
// 3D Structural Similarity Index Measure (SSIM) with windowed computation

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>

struct vec3i {
  int x, y, z;
  vec3i() : x(0), y(0), z(0) {}
  vec3i(int x, int y, int z) : x(x), y(y), z(z) {}
  vec3i operator+(const vec3i &o) const { return vec3i(x+o.x, y+o.y, z+o.z); }
  vec3i operator-(const vec3i &o) const { return vec3i(x-o.x, y-o.y, z-o.z); }
  int64_t long_product() const { return (int64_t)x * y * z; }
};

inline vec3i min3(const vec3i &a, const vec3i &b) {
  return vec3i(a.x<b.x?a.x:b.x, a.y<b.y?a.y:b.y, a.z<b.z?a.z:b.z);
}

template<typename T>
inline T next_multiple(T val, T divisor) {
  return ((val + divisor - 1) / divisor) * divisor;
}

#pragma omp declare target
template<int WIN_SIZE>
static void compute_ssim_kernel(
    uint32_t dimx, uint32_t dimy, uint32_t dimz,
    const float * __restrict__ fx, const float * __restrict__ fy,
    int gdims_x, int gdims_y, int /*gdims_z*/,
    float data_range, float cov_norm, float K1, float K2,
    float * __restrict__ out,
    int x, int y, int z)
{
  float ux=0, uy=0, uxx=0, uyy=0, uxy=0;
  for (int kz=0; kz<WIN_SIZE; kz++)
  for (int ky=0; ky<WIN_SIZE; ky++)
  for (int kx=0; kx<WIN_SIZE; kx++) {
    int gx = x+kx, gy = y+ky, gz = z+kz;
    uint32_t gidx = gx + gy * gdims_x + gz * gdims_x * gdims_y;
    float fxv = fx[gidx], fyv = fy[gidx];
    ux  += fxv;  uy  += fyv;
    uxx += fxv*fxv; uyy += fyv*fyv; uxy += fxv*fyv;
  }
  float w = 1.0f / (WIN_SIZE*WIN_SIZE*WIN_SIZE);
  ux*=w; uy*=w; uxx*=w; uyy*=w; uxy*=w;

  float vx  = cov_norm * (uxx - ux*ux);
  float vy  = cov_norm * (uyy - uy*uy);
  float vxy = cov_norm * (uxy - ux*uy);

  float R=data_range;
  float C1=(K1*R)*(K1*R), C2=(K2*R)*(K2*R);
  float A1=2.0f*ux*uy+C1, A2=2.0f*vxy+C2;
  float B1=ux*ux+uy*uy+C1, B2=vx+vy+C2;
  float D=B1*B2;
  float S=(A1*A2)/D;
  out[x + y*(int)dimx + z*(int)dimx*(int)dimy] = S;
}
#pragma omp end declare target

int main() {
  const vec3i dims(4096, 1024, 1024);
  constexpr float K1 = 0.01f;
  constexpr float K2 = 0.03f;
  const float data_range = 1.0f;
  constexpr bool use_sample_covariance = true;
  constexpr int win_size = 7;
  constexpr int crop = win_size >> 1;
  constexpr int NP = win_size * win_size * win_size;
  constexpr float cov_norm = use_sample_covariance ? (float)NP / (NP-1) : 1.0f;

  const vec3i batch = min3(vec3i(4096,16,16), dims);
  const vec3i batch_grid = batch + vec3i(win_size-1, win_size-1, win_size-1);
  const size_t batch_grid_count = (size_t)next_multiple<int64_t>(batch_grid.long_product(), 256);

  const size_t input_size  = sizeof(float) * batch_grid_count;
  const size_t output_size = input_size * 3;

  float *h_grid_reference = (float*)malloc(input_size);
  float *h_grid_inference = (float*)malloc(input_size);

  srand(123);
  for (size_t i = 0; i < batch_grid_count; i++) {
    h_grid_reference[i] = rand() / (float)RAND_MAX;
    h_grid_inference[i] = h_grid_reference[i] * 0.75f;
  }

  size_t out_count = output_size / sizeof(float);
  float *grid_output    = (float*)malloc(output_size);
  float *grid_inference = h_grid_inference;
  float *grid_reference = h_grid_reference;

  #pragma omp target enter data \
    map(to: grid_inference[0:batch_grid_count], grid_reference[0:batch_grid_count]) \
    map(alloc: grid_output[0:out_count])

  float ssim_sum = 0.0f;
  long etime = 0;

  for (int bz = crop; bz < dims.z - crop; bz += batch.z)
  for (int by = crop; by < dims.y - crop; by += batch.y)
  for (int bx = crop; bx < dims.x - crop; bx += batch.x) {
    vec3i block = min3(batch, dims - vec3i(crop,crop,crop) - vec3i(bx,by,bz));
    int64_t block_count = block.long_product();
    if (block_count <= 0) continue;

    vec3i block_grid = block + vec3i(win_size-1,win_size-1,win_size-1);

    auto start = std::chrono::steady_clock::now();

    int bx_=block.x, by_=block.y, bz_=block.z;
    int bgx=block_grid.x, bgy=block_grid.y, bgz=block_grid.z;

    // compute SSIM for each output voxel
    #pragma omp target teams distribute parallel for collapse(3) thread_limit(256) \
      map(tofrom: ssim_sum)
    for (int z = 0; z < bz_; z++)
    for (int y = 0; y < by_; y++)
    for (int x = 0; x < bx_; x++) {
      compute_ssim_kernel<7>((uint32_t)bx_, (uint32_t)by_, (uint32_t)bz_,
                             grid_reference, grid_inference,
                             bgx, bgy, bgz,
                             data_range, cov_norm, K1, K2,
                             grid_output, x, y, z);
    }

    // Reduce output
    float local_sum = 0.0f;
    #pragma omp target teams distribute parallel for reduction(+:local_sum) thread_limit(256)
    for (int64_t i = 0; i < block_count; i++) local_sum += grid_output[i];
    ssim_sum += local_sum;

    auto end = std::chrono::steady_clock::now();
    etime += std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  }

  #pragma omp target exit data map(delete: grid_inference[0:batch_grid_count], \
                                           grid_reference[0:batch_grid_count], \
                                           grid_output[0:out_count])

  free(h_grid_reference); free(h_grid_inference); free(grid_output);

  printf("Total kernel execution time (s): %lf\n", etime * 1e-9);
  printf("Structural Similarity Index Measure: %f\n",
         ssim_sum / (dims - vec3i(win_size-1,win_size-1,win_size-1)).long_product());
  return 0;
}

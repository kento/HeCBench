// Daphne (points2image) benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <chrono>

static constexpr int IMG_W = 800;
static constexpr int IMG_H = 600;
static constexpr int PLANE = IMG_H * IMG_W;

struct Mat33 { double data[3][3]; };
struct Mat13 { double data[3]; };
struct Vec5  { double data[5]; };
struct Point4f { float x, y, z, intensity; };

static void compute_inv_extrinsic(const double E[4][4], Mat33& invR, Mat13& invT)
{
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      invR.data[r][c] = E[c][r];
  for (int r = 0; r < 3; r++) {
    invT.data[r] = 0.0;
    for (int c = 0; c < 3; c++)
      invT.data[r] -= invR.data[r][c] * E[c][3];
  }
}

#pragma omp declare target
void atomic_float_min(float* addr, float val) {
  float assumed = *addr;
  while (val < assumed) {
    float old;
    #pragma omp atomic compare capture
    { old = *addr; if (*addr == assumed) *addr = val; }
    if (old == assumed) break;
    assumed = old;
  }
}
#pragma omp end declare target

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <num_points> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_points = atoi(argv[1]);
  const int repeat     = atoi(argv[2]);

  Mat33 cam{};
  cam.data[0][0] = 500.0; cam.data[0][2] = 400.0;
  cam.data[1][1] = 500.0; cam.data[1][2] = 300.0;
  cam.data[2][2] = 1.0;

  Vec5 dist{};
  dist.data[0] = -0.198; dist.data[1] = 0.028;
  dist.data[2] = 0.0;    dist.data[3] = 0.0; dist.data[4] = 0.0;

  double E[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  Mat33 invR{}; Mat13 invT{};
  compute_inv_extrinsic(E, invR, invT);

  Point4f* points  = (Point4f*)malloc(num_points * sizeof(Point4f));
  float*   result  = (float*)  malloc(4 * PLANE  * sizeof(float));

  srand(42);
  const double fx = cam.data[0][0], fy = cam.data[1][1];
  const double cx = cam.data[0][2], cy = cam.data[1][2];
  for (int i = 0; i < num_points; i++) {
    float z = 3.f + 100.f * ((float)rand() / RAND_MAX);
    float x = (float)((cx / fx) * z * (2.0 * rand()/RAND_MAX - 1.0) * 1.1);
    float y = (float)((cy / fy) * z * (2.0 * rand()/RAND_MAX - 1.0) * 1.1);
    points[i].x = x; points[i].y = y; points[i].z = z;
    points[i].intensity = (float)rand() / RAND_MAX;
  }

  // Copy camera parameters as flat arrays for device
  double invR_d[9], invT_d[3], cam_d[9], dist_d[5];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) invR_d[r*3+c] = invR.data[r][c];
  for (int r = 0; r < 3; r++) invT_d[r] = invT.data[r];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) cam_d[r*3+c] = cam.data[r][c];
  for (int i = 0; i < 5; i++) dist_d[i] = dist.data[i];

  const int result_size = 4 * PLANE;

  #pragma omp target enter data map(alloc: points[0:num_points], result[0:result_size], \
      invR_d[0:9], invT_d[0:3], cam_d[0:9], dist_d[0:5])
  #pragma omp target update to(points[0:num_points], invR_d[0:9], invT_d[0:3], cam_d[0:9], dist_d[0:5])

  // Warm-up
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < result_size; i++) result[i] = 0.f;
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int idx = 0; idx < num_points; idx++) {
    Point4f p = points[idx];
    double pt[3];
    for (int r = 0; r < 3; r++) {
      pt[r] = invT_d[r];
      pt[r] += invR_d[r*3+0] * (double)p.x + invR_d[r*3+1] * (double)p.y + invR_d[r*3+2] * (double)p.z;
    }
    if (pt[2] <= 2.5) continue;
    double xn = pt[0] / pt[2];
    double yn = pt[1] / pt[2];
    double r2 = xn*xn + yn*yn, r4 = r2*r2, r6 = r4*r2;
    double radial = 1.0 + dist_d[0]*r2 + dist_d[1]*r4 + dist_d[4]*r6;
    double xd = xn*radial + 2.0*dist_d[2]*xn*yn + dist_d[3]*(r2 + 2.0*xn*xn);
    double yd = yn*radial + dist_d[2]*(r2 + 2.0*yn*yn) + 2.0*dist_d[3]*xn*yn;
    double u = cam_d[0]*xd + cam_d[2];
    double v = cam_d[4]*yd + cam_d[5];
    int px = (int)(u + 0.5), py = (int)(v + 0.5);
    if (px < 0 || px >= IMG_W || py < 0 || py >= IMG_H) continue;
    int pid = py * IMG_W + px;
    float cm_depth = (float)(pt[2] * 100.0);
    atomic_float_min(result + pid, cm_depth);
    if (result[pid] >= cm_depth) result[pid + PLANE] = p.intensity;
    result[pid + 2*PLANE] = -1.25f;
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < result_size; i++) result[i] = 0.f;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < num_points; idx++) {
      Point4f p = points[idx];
      double pt[3];
      for (int rr = 0; rr < 3; rr++) {
        pt[rr] = invT_d[rr];
        pt[rr] += invR_d[rr*3+0]*(double)p.x + invR_d[rr*3+1]*(double)p.y + invR_d[rr*3+2]*(double)p.z;
      }
      if (pt[2] <= 2.5) continue;
      double xn = pt[0]/pt[2], yn = pt[1]/pt[2];
      double r2 = xn*xn+yn*yn, r4 = r2*r2, r6 = r4*r2;
      double radial = 1.0 + dist_d[0]*r2 + dist_d[1]*r4 + dist_d[4]*r6;
      double xd = xn*radial + 2.0*dist_d[2]*xn*yn + dist_d[3]*(r2+2.0*xn*xn);
      double yd = yn*radial + dist_d[2]*(r2+2.0*yn*yn) + 2.0*dist_d[3]*xn*yn;
      double u = cam_d[0]*xd + cam_d[2];
      double v = cam_d[4]*yd + cam_d[5];
      int px = (int)(u+0.5), py = (int)(v+0.5);
      if (px < 0 || px >= IMG_W || py < 0 || py >= IMG_H) continue;
      int pid = py*IMG_W + px;
      float cm_depth = (float)(pt[2]*100.0);
      atomic_float_min(result + pid, cm_depth);
      if (result[pid] >= cm_depth) result[pid+PLANE] = p.intensity;
      result[pid+2*PLANE] = -1.25f;
    }
  }
  auto t1 = std::chrono::steady_clock::now();

  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
  double throughput = (double)num_points / (elapsed_ms * 1e-3) / 1e6;
  printf("Average time per iteration: %.4f ms\n", elapsed_ms);
  printf("Throughput: %.2f M points/s\n", throughput);

  #pragma omp target update from(result[0:result_size])
  int   hit_count = 0;
  float dist_sum  = 0.f;
  for (int i = 0; i < PLANE; i++) {
    float d = result[i];
    if (d > 0.f) { hit_count++; dist_sum += d; }
  }
  printf("Pixels hit: %d / %d\n", hit_count, PLANE);
  printf("Distance sum: %.2f cm\n", dist_sum);
  printf("%s\n", (hit_count > 0) ? "PASS" : "FAIL");

  #pragma omp target exit data map(delete: points[0:num_points], result[0:result_size], \
      invR_d[0:9], invT_d[0:3], cam_d[0:9], dist_d[0:5])
  free(points); free(result);
  return 0;
}

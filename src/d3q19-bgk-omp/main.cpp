// D3Q19 BGK LBM benchmark – OpenMP target offloading port
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#pragma omp declare target
int d3q19_cx(int i) {
  const int v[19] = {0,-1,0,0,-1,-1,-1,-1,0,0,1,0,0,1,1,1,1,0,0};
  return v[i];
}
int d3q19_cy(int i) {
  const int v[19] = {0,0,-1,0,-1,1,0,0,-1,-1,0,1,0,1,-1,0,0,1,1};
  return v[i];
}
int d3q19_cz(int i) {
  const int v[19] = {0,0,0,-1,0,0,-1,1,-1,1,0,0,1,0,0,1,-1,1,-1};
  return v[i];
}
int d3q19_opp(int i) {
  const int v[19] = {0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9};
  return v[i];
}
double d3q19_w(int i) {
  if (i == 0) return 1.0 / 3.0;
  if (i >= 1 && i <= 3)  return 1.0 / 18.0;
  if (i >= 4 && i <= 9)  return 1.0 / 36.0;
  if (i >= 10 && i <= 12) return 1.0 / 18.0;
  return 1.0 / 36.0;
}

double d3q19_feq(int i, double rho, double ux, double uy, double uz) {
  double cu  = d3q19_cx(i) * ux + d3q19_cy(i) * uy + d3q19_cz(i) * uz;
  double usq = ux * ux + uy * uy + uz * uz;
  return d3q19_w(i) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
}

void macroscopic(const double f[19],
                 double& rho, double& ux, double& uy, double& uz) {
  double xm = f[1]+f[4]+f[5]+f[6]+f[7];
  double xp = f[10]+f[13]+f[14]+f[15]+f[16];
  double ym = f[2]+f[4]+f[8]+f[9]+f[14];
  double yp = f[5]+f[11]+f[13]+f[17]+f[18];
  double zm = f[3]+f[6]+f[8]+f[16]+f[18];
  double zp = f[7]+f[9]+f[12]+f[15]+f[17];
  rho = xm + xp + f[0]+f[2]+f[3]+f[8]+f[9]+f[11]+f[12]+f[17]+f[18];
  double inv = 1.0 / rho;
  ux = (xp - xm) * inv;
  uy = (yp - ym) * inv;
  uz = (zp - zm) * inv;
}

int idf(int x, int y, int z, int i, int nx, int ny, int nz) {
  return i * (nz * ny * nx) + z * (ny * nx) + y * nx + x;
}
#pragma omp end declare target

void collide_and_stream(
    double* f0, double* f1,
    int nx, int ny, int nz,
    double omega, double ulb)
{
  #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
  for (int z = 1; z < nz-1; z++) {
    for (int y = 1; y < ny-1; y++) {
      for (int x = 1; x < nx-1; x++) {
        double fin[19];
        for (int i = 0; i < 19; i++) {
          int xs = x - d3q19_cx(i);
          int ys = y - d3q19_cy(i);
          int zs = z - d3q19_cz(i);
          bool is_wall = (xs < 1 || xs > nx-2 || ys < 1 || ys > ny-2 || zs < 1 || zs > nz-2);
          if (!is_wall) {
            fin[i] = f0[idf(xs, ys, zs, i, nx, ny, nz)];
          } else {
            fin[i] = f0[idf(x, y, z, d3q19_opp(i), nx, ny, nz)];
            if (zs == 0) {
              int cxi = d3q19_cx(i);
              if      (cxi == -1) fin[i] -= 2.0 * (1.0/36.0) * 3.0 * ulb;
              else if (cxi ==  1) fin[i] += 2.0 * (1.0/36.0) * 3.0 * ulb;
            }
          }
        }
        double rho, ux, uy, uz;
        macroscopic(fin, rho, ux, uy, uz);
        for (int i = 0; i < 19; i++) {
          double feq = d3q19_feq(i, rho, ux, uy, uz);
          f1[idf(x, y, z, i, nx, ny, nz)] = (1.0 - omega) * fin[i] + omega * feq;
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    printf("Usage: %s <nx> <ny> <nz> <iterations>\n", argv[0]);
    return 1;
  }
  const int nx         = atoi(argv[1]);
  const int ny         = atoi(argv[2]);
  const int nz         = atoi(argv[3]);
  const int iterations = atoi(argv[4]);
  const int nl         = nx * ny * nz;

  const double ulb   = 0.02;
  const double nu    = ulb * (ny - 2) / 100.0;
  const double omega = 1.0 / (3.0 * nu + 0.5);
  printf("nx=%d ny=%d nz=%d  omega=%.6f\n", nx, ny, nz, omega);

  double* f0 = (double*)malloc(19 * nl * sizeof(double));
  double* f1 = (double*)malloc(19 * nl * sizeof(double));

  #pragma omp target enter data map(alloc: f0[0:19*nl], f1[0:19*nl])

  // Initialise
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int flat = 0; flat < nl; flat++) {
    int x = flat % nx;
    int y = (flat / nx) % ny;
    int z = flat / (nx * ny);
    for (int i = 0; i < 19; i++)
      f0[idf(x, y, z, i, nx, ny, nz)] = d3q19_feq(i, 1.0, 0.0, 0.0, 0.0);
  }

  const int warmup = 100;
  for (int t = 0; t < warmup; t++) {
    collide_and_stream(f0, f1, nx, ny, nz, omega, ulb);
    // Swap f0 ↔ f1 via copy on device
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < 19*nl; i++) f0[i] = f1[i];
  }
  printf("Warmup (%d iters) done.\n", warmup);

  auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < iterations; t++) {
    collide_and_stream(f0, f1, nx, ny, nz, omega, ulb);
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < 19*nl; i++) f0[i] = f1[i];
  }
  auto t1 = std::chrono::steady_clock::now();

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  double mlups   = ((double)nl * iterations / elapsed) / 1.0e6;
  printf("Time: %.3f s  MLUPS: %.2f\n", elapsed, mlups);

  double pop_sum = 0.0;
  #pragma omp target teams distribute parallel for reduction(+:pop_sum) thread_limit(256)
  for (int flat = 0; flat < nl; flat++) {
    int x = flat % nx;
    int y = (flat / nx) % ny;
    int z = flat / (nx * ny);
    for (int i = 0; i < 19; i++)
      pop_sum += f1[idf(x, y, z, i, nx, ny, nz)];
  }
  printf("Population sum: %.6f  (expected ≈ %d)\n", pop_sum, nl);

  #pragma omp target exit data map(delete: f0[0:19*nl], f1[0:19*nl])
  free(f0); free(f1);
  return 0;
}

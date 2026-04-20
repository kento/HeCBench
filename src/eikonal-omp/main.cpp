// OpenMP target offloading port of eikonal-kokkos (Fast Iterative Method for Eikonal)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>

#define INF     1e20

#pragma omp declare target
static double eikonalUpdate(
    const double* sol,
    double speed,
    int i, int j, int k,
    int nx, int ny, int nz)
{
  auto idx = [&](int x, int y, int z) -> int {
    return x + y * nx + z * nx * ny;
  };

  double xa = (i > 0)      ? sol[idx(i-1,j,k)] : INF;
  double xb = (i < nx-1)   ? sol[idx(i+1,j,k)] : INF;
  double ya = (j > 0)      ? sol[idx(i,j-1,k)] : INF;
  double yb = (j < ny-1)   ? sol[idx(i,j+1,k)] : INF;
  double za = (k > 0)      ? sol[idx(i,j,k-1)] : INF;
  double zb = (k < nz-1)   ? sol[idx(i,j,k+1)] : INF;

  double mx = fmin(xa, xb);
  double my = fmin(ya, yb);
  double mz = fmin(za, zb);

  if (mx > my) { double t = mx; mx = my; my = t; }
  if (my > mz) { double t = my; my = mz; mz = t; }
  if (mx > my) { double t = mx; mx = my; my = t; }

  double f = 1.0 / speed;

  double u = mx + f;
  if (u <= my) return u;

  double disc2 = 2.0 * f * f - (my - mx) * (my - mx);
  if (disc2 >= 0.0) {
    u = 0.5 * (mx + my + sqrt(disc2));
    if (u <= mz) return u;
  }

  double s = mx + my + mz;
  double disc3 = s * s - 3.0 * (mx*mx + my*my + mz*mz - f*f);
  if (disc3 >= 0.0) {
    return (s + sqrt(disc3)) / 3.0;
  }

  return mz + f;
}
#pragma omp end declare target

int main(int argc, char** argv)
{
  size_t size = 64;
  size_t itersPerBlock = 10;
  size_t type = 0;
  std::string name = "output.nrrd";
  bool verbose = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-s") == 0 && i+1 < argc) size = atoi(argv[++i]);
    if (strcmp(argv[i], "-m") == 0 && i+1 < argc) type = atoi(argv[++i]);
    if (strcmp(argv[i], "-i") == 0 && i+1 < argc) itersPerBlock = atoi(argv[++i]);
    if (strcmp(argv[i], "-o") == 0 && i+1 < argc) name = argv[++i];
    if (strcmp(argv[i], "-v") == 0)                verbose = true;
  }

  int nx = (int)size, ny = (int)size, nz = (int)size;
  int N = nx * ny * nz;

  printf("Volume: %d x %d x %d = %d voxels\n", nx, ny, nz, N);

  double* d_sol = (double*)malloc(N * sizeof(double));
  double* d_spd = (double*)malloc(N * sizeof(double));

  for (int i = 0; i < N; i++) { d_sol[i] = INF; d_spd[i] = 1.0; }
  d_sol[0] = 0.0;

  #pragma omp target enter data map(tofrom: d_sol[0:N], d_spd[0:N])

  if (type == 1) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int idx = 0; idx < N; idx++) {
      int z = idx / (nx * ny);
      int r = idx % (nx * ny);
      int y = r / nx, x = r % nx;
      double fx = (double)x / nx * 2.0 * 3.14159265358979;
      double fy = (double)y / ny * 2.0 * 3.14159265358979;
      double fz = (double)z / nz * 2.0 * 3.14159265358979;
      d_spd[idx] = 0.5 + 0.5 * sin(fx) * sin(fy) * sin(fz);
    }
  }

  auto t0 = std::chrono::steady_clock::now();

  bool changed = true;
  int totalIter = 0;
  while (changed) {
    changed = false;
    int nChanged = 0;

    for (int innerIt = 0; innerIt < (int)itersPerBlock; innerIt++) {
      int localChanged = 0;
      #pragma omp target teams distribute parallel for reduction(+:localChanged) thread_limit(256)
      for (int idx = 0; idx < N; idx++) {
        int z = idx / (nx * ny);
        int r = idx % (nx * ny);
        int y = r / nx, x = r % nx;

        double oldVal = d_sol[idx];
        double newVal = eikonalUpdate(d_sol, d_spd[idx], x, y, z, nx, ny, nz);
        if (newVal < oldVal) newVal = oldVal;

        if (fabs(newVal - oldVal) > 1e-12) {
          d_sol[idx] = newVal;
          localChanged++;
        }
      }
      nChanged += localChanged;
    }

    if (nChanged > 0) changed = true;
    totalIter++;

    if (verbose)
      printf("Iteration %d: %d voxels updated\n", totalIter, nChanged);

    if (totalIter > nx + ny + nz) break;
  }

  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
  printf("Solver finished in %d outer iterations, %.2f ms\n", totalIter, ms);

  #pragma omp target update from(d_sol[0:N])

  double checksum = 0.0;
  for (int i = 0; i < N; i++)
    checksum += (d_sol[i] < INF ? d_sol[i] : 0.0);
  printf("Checksum = %lf\n", checksum / N);

  std::fstream out(name.c_str(), std::ios::out | std::ios::binary);
  out << "NRRD0001\n# OMP eikonal output\n"
      << "type: double\ndimension: 3\n"
      << "sizes: " << nx << " " << ny << " " << nz << "\n"
      << "endian: little\nencoding: raw\n\n";
  for (int i = 0; i < N; i++)
    out.write(reinterpret_cast<const char*>(&d_sol[i]), sizeof(double));
  out.close();
  printf("Wrote %s\n", name.c_str());

  #pragma omp target exit data map(delete: d_sol[0:N], d_spd[0:N])
  free(d_sol);
  free(d_spd);
  return 0;
}

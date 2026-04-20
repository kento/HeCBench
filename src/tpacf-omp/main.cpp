// OpenMP target offloading port of tpacf benchmark
// Two-Point Angular Correlation Function

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#define NUMBINS 64
#define D2R (M_PI / 180.0)

static double* init_bins(int bins_per_dec, float min_angle, float max_angle,
                         int angle_units, int *nbins) {
  *nbins = (int)floor(bins_per_dec * (log10(max_angle) - log10(min_angle)));
  static double binb[31];
  int binoffset = 30 - (*nbins);
  for (int k = 0; k < (*nbins)+1; k++) {
    double bb = pow(10.0, log10(min_angle) + k * 1.0 / bins_per_dec);
    binb[k + binoffset] = cos(bb / (angle_units ? 60.0 : 1.0) * D2R);
  }
  for (int k = 0; k < binoffset; k++) binb[k] = -5.0;
  binb[30] = -5.0;
  return binb;
}

struct CartesianData {
  std::vector<double> x, y, z;
  int n;
  CartesianData(int n_) : n(n_), x(n_), y(n_), z(n_) {}
};

static CartesianData generate_data(int n, unsigned seed) {
  CartesianData cd(n);
  srand(seed);
  for (int i = 0; i < n; i++) {
    double ra  = (rand() / (double)RAND_MAX) * 360.0;
    double dec = (rand() / (double)RAND_MAX) * 180.0 - 90.0;
    double ra_r  = ra  * D2R;
    double dec_r = dec * D2R;
    cd.x[i] = cos(dec_r) * cos(ra_r);
    cd.y[i] = cos(dec_r) * sin(ra_r);
    cd.z[i] = sin(dec_r);
  }
  return cd;
}

static void compute_histogram(
    const double *x1, const double *y1, const double *z1,
    const double *x2, const double *y2, const double *z2,
    int n1, int n2,
    long long *histo,
    const double *d_binb, int nbins)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < n1; i++) {
    double xi = x1[i], yi = y1[i], zi = z1[i];
    for (int j = 0; j < n2; j++) {
      double dot = xi * x2[j] + yi * y2[j] + zi * z2[j];
      if (dot >  1.0) dot =  1.0;
      if (dot < -1.0) dot = -1.0;
      int bin = nbins - 1;
      for (int b = 0; b < nbins - 1; b++) {
        if (dot > d_binb[b]) { bin = b; break; }
      }
      #pragma omp atomic
      histo[bin]++;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    printf("Usage: %s <data_size> <random_size> <random_count> <repeat>\n", argv[0]);
    return 1;
  }
  int nd = atoi(argv[1]);
  int nr = atoi(argv[2]);
  int repeat = atoi(argv[4]);

  nd -= nd % 4;
  nr -= nr % 4;

  const int bins_per_dec = 5;
  const float min_angle = 0.01f;
  const float max_angle = 10.0f;
  const int angle_units = 0;

  int nbins = 0;
  double *binb = init_bins(bins_per_dec, min_angle, max_angle, angle_units, &nbins);

  printf("\ndata size: %d\n", nd);
  printf("random size: %d\n", nr);
  printf("nbins: %d\n\n", nbins);

  CartesianData data   = generate_data(nd, 1);
  CartesianData random = generate_data(nr, 2);

  double *dx = data.x.data(),   *dy = data.y.data(),   *dz = data.z.data();
  double *rx = random.x.data(), *ry = random.y.data(), *rz = random.z.data();

  std::vector<long long> h_DD(NUMBINS, 0), h_DR(NUMBINS, 0), h_RR(NUMBINS, 0);
  long long *d_DD = h_DD.data(), *d_DR = h_DR.data(), *d_RR = h_RR.data();

  #pragma omp target enter data map(to: dx[0:nd], dy[0:nd], dz[0:nd]) \
                                  map(to: rx[0:nr], ry[0:nr], rz[0:nr]) \
                                  map(to: binb[0:NUMBINS]) \
                                  map(alloc: d_DD[0:NUMBINS], d_DR[0:NUMBINS], d_RR[0:NUMBINS])

  auto t_start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // Zero histograms
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int b = 0; b < NUMBINS; b++) { d_DD[b] = 0; d_DR[b] = 0; d_RR[b] = 0; }

    compute_histogram(dx, dy, dz, dx, dy, dz, nd, nd, d_DD, binb, NUMBINS);
    compute_histogram(dx, dy, dz, rx, ry, rz, nd, nr, d_DR, binb, NUMBINS);
    compute_histogram(rx, ry, rz, rx, ry, rz, nr, nr, d_RR, binb, NUMBINS);
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
  printf("DONE! after %f\n", elapsed);

  #pragma omp target update from(d_DD[0:NUMBINS], d_DR[0:NUMBINS], d_RR[0:NUMBINS])
  #pragma omp target exit data map(delete: dx[0:nd], dy[0:nd], dz[0:nd], \
                                           rx[0:nr], ry[0:nr], rz[0:nr], \
                                           binb[0:NUMBINS], \
                                           d_DD[0:NUMBINS], d_DR[0:NUMBINS], d_RR[0:NUMBINS])

  printf("bin  DD         DR         RR\n");
  for (int b = 0; b < 10 && b < NUMBINS; b++)
    printf("%3d  %lld  %lld  %lld\n", b, h_DD[b], h_DR[b], h_RR[b]);

  return 0;
}

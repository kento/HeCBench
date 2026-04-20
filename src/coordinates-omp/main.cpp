// Coordinates (cuspatial) benchmark – OpenMP target offloading port
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

template <typename T>
struct lonlat_2d { T x, y; };

template <typename T>
struct cartesian_2d { T x, y; };

template <typename T>
struct to_cartesian_functor {
  T ox_rad, oy_rad;

  to_cartesian_functor(T lon_orig, T lat_orig) {
    constexpr T pi = T(3.14159265358979323846);
    ox_rad = lon_orig * pi / T(180);
    oy_rad = lat_orig * pi / T(180);
  }
};

#pragma omp declare target
template <typename T>
cartesian_2d<T> convert_coords(T ox_rad, T oy_rad, T p_lon, T p_lat) {
  constexpr T pi = T(3.14159265358979323846);
  T p_rad_x = p_lon * pi / T(180);
  T p_rad_y = p_lat * pi / T(180);
  cartesian_2d<T> result;
  result.x = (p_rad_x - ox_rad) * cos(oy_rad);
  result.y = p_rad_y - oy_rad;
  return result;
}
#pragma omp end declare target

template <typename T>
bool run(int num_coords, int repeat) {
  std::srand(42);

  T* h_lon = (T*)malloc(num_coords * sizeof(T));
  T* h_lat = (T*)malloc(num_coords * sizeof(T));
  for (int i = 0; i < num_coords; ++i) {
    h_lon[i] = T(-180) + T(360) * std::rand() / RAND_MAX;
    h_lat[i] = T(-90)  + T(180) * std::rand() / RAND_MAX;
  }

  const T orig_lon = T(-73.9857);
  const T orig_lat = T( 40.7484);
  constexpr T pi = T(3.14159265358979323846);
  const T ox_rad = orig_lon * pi / T(180);
  const T oy_rad = orig_lat * pi / T(180);

  // CPU reference
  T* h_ref_x = (T*)malloc(num_coords * sizeof(T));
  T* h_ref_y = (T*)malloc(num_coords * sizeof(T));
  for (int i = 0; i < num_coords; ++i) {
    auto r = convert_coords(ox_rad, oy_rad, h_lon[i], h_lat[i]);
    h_ref_x[i] = r.x;
    h_ref_y[i] = r.y;
  }

  T* d_lon  = (T*)malloc(num_coords * sizeof(T));
  T* d_lat  = (T*)malloc(num_coords * sizeof(T));
  T* d_outx = (T*)malloc(num_coords * sizeof(T));
  T* d_outy = (T*)malloc(num_coords * sizeof(T));
  for (int i = 0; i < num_coords; ++i) { d_lon[i] = h_lon[i]; d_lat[i] = h_lat[i]; }

  #pragma omp target enter data map(alloc: d_lon[0:num_coords], d_lat[0:num_coords], \
      d_outx[0:num_coords], d_outy[0:num_coords])
  #pragma omp target update to(d_lon[0:num_coords], d_lat[0:num_coords])

  // Warm-up
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < num_coords; i++) {
    auto r = convert_coords(ox_rad, oy_rad, d_lon[i], d_lat[i]);
    d_outx[i] = r.x;
    d_outy[i] = r.y;
  }

  for (int r = 0; r < repeat; ++r) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < num_coords; i++) {
      auto res = convert_coords(ox_rad, oy_rad, d_lon[i], d_lat[i]);
      d_outx[i] = res.x;
      d_outy[i] = res.y;
    }
  }

  #pragma omp target update from(d_outx[0:num_coords], d_outy[0:num_coords])

  constexpr T eps = (sizeof(T) == 4) ? T(1e-5) : T(1e-10);
  bool pass = true;
  for (int i = 0; i < num_coords; ++i) {
    if (std::fabs(d_outx[i] - h_ref_x[i]) > eps ||
        std::fabs(d_outy[i] - h_ref_y[i]) > eps) {
      printf("  MISMATCH at %d: got (%.10g, %.10g) expected (%.10g, %.10g)\n",
             i, (double)d_outx[i], (double)d_outy[i], (double)h_ref_x[i], (double)h_ref_y[i]);
      pass = false;
      break;
    }
  }

  #pragma omp target exit data map(delete: d_lon[0:num_coords], d_lat[0:num_coords], \
      d_outx[0:num_coords], d_outy[0:num_coords])
  free(d_lon); free(d_lat); free(d_outx); free(d_outy);
  free(h_lon); free(h_lat); free(h_ref_x); free(h_ref_y);
  return pass;
}

int main(int argc, char* argv[]) {
  int num_coords = (argc > 1) ? std::atoi(argv[1]) : 100000;
  int repeat     = (argc > 2) ? std::atoi(argv[2]) : 100;

  printf("coordinates: num_coords=%d  repeat=%d\n", num_coords, repeat);

  bool ok_d = run<double>(num_coords, repeat);
  printf("double: %s\n", ok_d ? "PASS" : "FAIL");

  bool ok_f = run<float>(num_coords, repeat);
  printf("float:  %s\n", ok_f ? "PASS" : "FAIL");
  return 0;
}

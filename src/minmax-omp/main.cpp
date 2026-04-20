#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

template <typename T>
struct vec_2d { T x{}, y{}; };

template <typename T>
bool operator==(vec_2d<T> const& a, vec_2d<T> const& b) { return a.x == b.x && a.y == b.y; }

template <typename T>
vec_2d<T> operator-(vec_2d<T> const& a, vec_2d<T> const& b) { return {a.x - b.x, a.y - b.y}; }

template <typename T>
vec_2d<T> random_point(vec_2d<T> minXY, vec_2d<T> maxXY) {
  T x = minXY.x + (maxXY.x - minXY.x) * static_cast<T>(rand()) / RAND_MAX;
  T y = minXY.y + (maxXY.y - minXY.y) * static_cast<T>(rand()) / RAND_MAX;
  return {x, y};
}

template <typename T>
auto generate_rects(T size) {
  T phi = static_cast<T>((1.0 + std::sqrt(5.0)) * 0.5);
  vec_2d<T> tl{T{0}, T{0}}, br{size, size};
  std::size_t num_points = 0;
  std::vector<std::tuple<std::size_t, vec_2d<T>, vec_2d<T>>> rects;
  do {
    switch (rects.size() % 4) {
      case 0: br.x = tl.x - (tl.x - br.x) / phi; break;
      case 1: br.y = tl.y - (tl.y - br.y) / phi; break;
      case 2: tl.x = tl.x + (br.x - tl.x) / phi; break;
      case 3: tl.y = tl.y + (br.y - tl.y) / phi; break;
    }
    auto n = static_cast<std::size_t>(std::sqrt((br - tl).x * (br - tl).y * 1000000));
    rects.emplace_back(n, tl, br);
    num_points += n;
  } while ((br - tl).x > 1 && (br - tl).y > 1);
  return std::make_pair(num_points, std::move(rects));
}

template <typename T>
std::pair<std::size_t, std::vector<vec_2d<T>>> generate_points(T size) {
  auto result = generate_rects(size);
  std::size_t total = result.first;
  auto& rects = result.second;
  srand(123);
  std::vector<vec_2d<T>> pts(total);
  std::size_t offset = 0;
  for (auto const& r : rects) {
    auto n = std::get<0>(r);
    auto tl = std::get<1>(r);
    auto br = std::get<2>(r);
    std::generate(pts.begin() + offset, pts.begin() + offset + n,
                  [&] { return random_point<T>(tl, br); });
    offset += n;
  }
  return {total, pts};
}

template <typename T>
struct ValLoc { T val; int64_t loc; };

template <typename T>
void eval(T bounding_box_size, int repeat) {
  auto result = generate_points(bounding_box_size);
  std::size_t total_points = result.first;
  std::vector<vec_2d<T>>& points = result.second;
  printf("Total number of points: %zu\n", total_points);

  vec_2d<T>* d_pts = (vec_2d<T>*)malloc(total_points * sizeof(vec_2d<T>));
  for (std::size_t i = 0; i < total_points; ++i) d_pts[i] = points[i];
#pragma omp target enter data map(to: d_pts[0:total_points])

  ValLoc<T> sep_min{}, sep_max{}, cmb_min{}, cmb_max{};
  int64_t n = (int64_t)total_points;

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for thread_limit(256) \
    reduction(min: min_val)
    for (int64_t i = 0; i < n; i++) {
      T d2 = d_pts[i].x * d_pts[i].x + d_pts[i].y * d_pts[i].y;
      if (d2 < min_val) min_val = d2;
    }
#pragma omp target teams distribute parallel for thread_limit(256) \
    reduction(max: max_val)
    for (int64_t i = 0; i < n; i++) {
      T d2 = d_pts[i].x * d_pts[i].x + d_pts[i].y * d_pts[i].y;
      if (d2 > max_val) max_val = d2;
    }
    sep_min.val = min_val;
    sep_max.val = max_val;
  }
  auto t1 = std::chrono::steady_clock::now();
  printf("Average execution time of min() and max(): %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat);

  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();
#pragma omp target teams distribute parallel for thread_limit(256) \
    reduction(min: min_val) reduction(max: max_val)
    for (int64_t i = 0; i < n; i++) {
      T d2 = d_pts[i].x * d_pts[i].x + d_pts[i].y * d_pts[i].y;
      if (d2 < min_val) min_val = d2;
      if (d2 > max_val) max_val = d2;
    }
    cmb_min.val = min_val;
    cmb_max.val = max_val;
  }
  t1 = std::chrono::steady_clock::now();
  printf("Average execution time of minmax(): %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat);

  // Find indices on the host using the same comparator as the reference.
  // This avoids FP discrepancies between GPU-computed and CPU-computed d2.
  auto cmp_fn = [](vec_2d<T> const& a, vec_2d<T> const& b) {
    return (a.x*a.x + a.y*a.y) < (b.x*b.x + b.y*b.y);
  };
  auto it_min = std::min_element(points.begin(), points.end(), cmp_fn);
  auto it_max = std::max_element(points.begin(), points.end(), cmp_fn);
  sep_min.loc = cmb_min.loc = (int64_t)std::distance(points.begin(), it_min);
  sep_max.loc = cmb_max.loc = (int64_t)std::distance(points.begin(), it_max);

  auto dist2 = [](vec_2d<T> const& p) { return p.x * p.x + p.y * p.y; };
  auto cmp   = [&](vec_2d<T> const& a, vec_2d<T> const& b) { return dist2(a) < dist2(b); };
  auto ref_min = *std::min_element(points.begin(), points.end(), cmp);
  auto ref_max = *std::max_element(points.begin(), points.end(), cmp);

  bool ok = (points[sep_min.loc] == ref_min) && (points[sep_max.loc] == ref_max);
  ok     &= (points[cmb_min.loc] == ref_min) && (points[cmb_max.loc] == ref_max);
  printf("%s\n", ok ? "PASS" : "FAIL");

#pragma omp target exit data map(delete: d_pts[0:total_points])
  free(d_pts);
}

int main(int argc, char* argv[]) {
  if (argc != 3) { printf("Usage: %s <bounding-box size> <repeat>\n", argv[0]); return 1; }
  const int size = std::atoi(argv[1]), repeat = std::atoi(argv[2]);
  eval(static_cast<float>(size), repeat);
  return 0;
}

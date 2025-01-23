#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, CUDA kernel.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// Template parameters are describe CTA size
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
template <int bx, int by>
void JacobiIteration(const float *du0, const float *dv0,
                                const float *Ix, const float *Iy,
                                const float *Iz, int w, int h, int s,
                                float alpha, float *du1, float *dv1,
                                const sycl::nd_item<3> &item,
                                volatile float *du, volatile float *dv) {
  // Handle to thread block group
  auto cta = item.get_group();

  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);

  // position within global memory array
  const int pos = sycl::min(ix, w - 1) + sycl::min(iy, h - 1) * s;

  // position within shared memory array
  const int shMemPos =
      item.get_local_id(2) + 1 + (item.get_local_id(1) + 1) * (bx + 2);

  // Load data to shared memory.
  // load tile being processed
  du[shMemPos] = du0[pos];
  dv[shMemPos] = dv0[pos];

  // load necessary neighbouring elements
  // We clamp out-of-range coordinates.
  // It is equivalent to mirroring
  // because we access data only one step away from borders.
  if (item.get_local_id(1) == 0) {
    // beginning of the tile
    const int bsx = item.get_group(2) * item.get_local_range(2);
    const int bsy = item.get_group(1) * item.get_local_range(1);
    // element position within matrix
    int x, y;
    // element position within linear array
    // gm - global memory
    // sm - shared memory
    int gmPos, smPos;

    x = sycl::min(bsx + (int)item.get_local_id(2), w - 1);
    // row just below the tile
    y = sycl::max(bsy - 1, 0);
    gmPos = y * s + x;
    smPos = item.get_local_id(2) + 1;
    du[smPos] = du0[gmPos];
    dv[smPos] = dv0[gmPos];

    // row above the tile
    y = sycl::min(bsy + by, h - 1);
    smPos += (by + 1) * (bx + 2);
    gmPos = y * s + x;
    du[smPos] = du0[gmPos];
    dv[smPos] = dv0[gmPos];
  } else if (item.get_local_id(1) == 1) {
    // beginning of the tile
    const int bsx = item.get_group(2) * item.get_local_range(2);
    const int bsy = item.get_group(1) * item.get_local_range(1);
    // element position within matrix
    int x, y;
    // element position within linear array
    // gm - global memory
    // sm - shared memory
    int gmPos, smPos;

    y = sycl::min(bsy + (int)item.get_local_id(2), h - 1);
    // column to the left
    x = sycl::max(bsx - 1, 0);
    smPos = bx + 2 + item.get_local_id(2) * (bx + 2);
    gmPos = x + y * s;

    // check if we are within tile
    if (item.get_local_id(2) < by) {
      du[smPos] = du0[gmPos];
      dv[smPos] = dv0[gmPos];
      // column to the right
      x = sycl::min(bsx + bx, w - 1);
      gmPos = y * s + x;
      smPos += bx + 1;
      du[smPos] = du0[gmPos];
      dv[smPos] = dv0[gmPos];
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (ix >= w || iy >= h) return;

  // now all necessary data are loaded to shared memory
  int left, right, up, down;
  left = shMemPos - 1;
  right = shMemPos + 1;
  up = shMemPos + bx + 2;
  down = shMemPos - bx - 2;

  float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
  float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

  float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) /
               (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

  du1[pos] = sumU - Ix[pos] * frac;
  dv1[pos] = sumV - Iy[pos] * frac;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief one iteration of classical Horn-Schunck method, CUDA kernel wrapper.
///
/// It is one iteration of Jacobi method for a corresponding linear system.
/// \param[in]  du0     current horizontal displacement approximation
/// \param[in]  dv0     current vertical displacement approximation
/// \param[in]  Ix      image x derivative
/// \param[in]  Iy      image y derivative
/// \param[in]  Iz      temporal derivative
/// \param[in]  w       width
/// \param[in]  h       height
/// \param[in]  s       stride
/// \param[in]  alpha   degree of smoothness
/// \param[out] du1     new horizontal displacement approximation
/// \param[out] dv1     new vertical displacement approximation
///////////////////////////////////////////////////////////////////////////////
static void SolveForUpdate(const float *du0, const float *dv0, const float *Ix,
                           const float *Iy, const float *Iz, int w, int h,
                           int s, float alpha, float *du1, float *dv1, sycl::queue &q) {
  // CTA size
  sycl::range<3> threads(1, 6, 32);
  // grid size
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> du_acc_ct1(
        sycl::range<1>((32 + 2) * (6 + 2)), cgh);
    sycl::local_accessor<float, 1> dv_acc_ct1(
        sycl::range<1>((32 + 2) * (6 + 2)), cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item) {
                       JacobiIteration<32, 6>(du0, dv0, Ix, Iy, Iz, w, h, s,
                                              alpha, du1, dv1, item,
                                              du_acc_ct1.get_pointer(),
                                              dv_acc_ct1.get_pointer());
                     });
  });
}

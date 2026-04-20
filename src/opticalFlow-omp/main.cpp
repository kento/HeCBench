// OpenMP target offloading port of Horn-Schunck optical flow benchmark.
// Coarse-to-fine pyramid with warping iterations.
// Translated from Kokkos MDRangePolicy<Rank<2>> version.

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>

// ============================================================
// Parameters
// ============================================================
static constexpr int   IMG_W          = 768;
static constexpr int   IMG_H          = 576;
static constexpr float ALPHA          = 0.2f;
static constexpr int   N_LEVELS       = 5;
static constexpr int   N_WARP_ITERS   = 3;
static constexpr int   N_SOLVER_ITERS = 500;
static constexpr int   IMG_SZ         = IMG_W * IMG_H;

// ============================================================
// Device-side helper (clamped integer index)
// ============================================================
#pragma omp declare target
static inline int iclamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
#pragma omp end declare target

// ============================================================
// Device memory helpers
// ============================================================
static float* devAlloc(int n) {
    float* p = (float*)malloc(n * sizeof(float));
    #pragma omp target enter data map(alloc: p[0:n])
    return p;
}

static void devFree(float* p, int n) {
    #pragma omp target exit data map(delete: p[0:n])
    free(p);
}

// Fill device array with zero
static void devZero(float* p, int n) {
    #pragma omp target teams distribute parallel for thread_limit(256) \
        map(alloc: p[0:n])
    for (int i = 0; i < n; i++) p[i] = 0.f;
}

// Device-side copy
static void devCopy(const float* src, float* dst, int n) {
    #pragma omp target teams distribute parallel for thread_limit(256) \
        map(alloc: src[0:n], dst[0:n])
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

// Device-side add:  dst[i] += src[i]
static void devAdd(float* dst, const float* src, int n) {
    #pragma omp target teams distribute parallel for thread_limit(256) \
        map(alloc: dst[0:n], src[0:n])
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

// ============================================================
// Image derivative computation (5-point stencil)
// ============================================================
static void computeDerivatives(const float* I0, const float* I1,
                                int w, int h, int sz,
                                float* Ix, float* Iy, float* Iz) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
        map(alloc: I0[0:sz], I1[0:sz], Ix[0:sz], Iy[0:sz], Iz[0:sz])
    for (int iy = 0; iy < h; iy++) {
        for (int ix = 0; ix < w; ix++) {
            const int pos = iy * w + ix;
            // x-derivative (average over I0 and I1)
            float t0 = (-I0[iy*w + iclamp(ix-2,0,w-1)]
                       + 8.f*I0[iy*w + iclamp(ix-1,0,w-1)]
                       - 8.f*I0[iy*w + iclamp(ix+1,0,w-1)]
                       +     I0[iy*w + iclamp(ix+2,0,w-1)]) / 12.f;
            float t1 = (-I1[iy*w + iclamp(ix-2,0,w-1)]
                       + 8.f*I1[iy*w + iclamp(ix-1,0,w-1)]
                       - 8.f*I1[iy*w + iclamp(ix+1,0,w-1)]
                       +     I1[iy*w + iclamp(ix+2,0,w-1)]) / 12.f;
            Ix[pos] = (t0 + t1) * 0.5f;
            // y-derivative
            t0 = (-I0[iclamp(iy-2,0,h-1)*w + ix]
                 + 8.f*I0[iclamp(iy-1,0,h-1)*w + ix]
                 - 8.f*I0[iclamp(iy+1,0,h-1)*w + ix]
                 +     I0[iclamp(iy+2,0,h-1)*w + ix]) / 12.f;
            t1 = (-I1[iclamp(iy-2,0,h-1)*w + ix]
                 + 8.f*I1[iclamp(iy-1,0,h-1)*w + ix]
                 - 8.f*I1[iclamp(iy+1,0,h-1)*w + ix]
                 +     I1[iclamp(iy+2,0,h-1)*w + ix]) / 12.f;
            Iy[pos] = (t0 + t1) * 0.5f;
            // temporal derivative
            Iz[pos] = I1[pos] - I0[pos];
        }
    }
}

// ============================================================
// Jacobi iteration for Horn-Schunck
// ============================================================
static void jacobiIteration(const float* du0, const float* dv0,
                              const float* Ix, const float* Iy, const float* Iz,
                              int w, int h, int sz, float alpha,
                              float* du1, float* dv1) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
        map(alloc: du0[0:sz], dv0[0:sz], Ix[0:sz], Iy[0:sz], Iz[0:sz], \
                   du1[0:sz], dv1[0:sz])
    for (int iy = 0; iy < h; iy++) {
        for (int ix = 0; ix < w; ix++) {
            const int pos   = iy * w + ix;
            const int left  = iy * w + iclamp(ix-1, 0, w-1);
            const int right = iy * w + iclamp(ix+1, 0, w-1);
            const int up    = iclamp(iy-1, 0, h-1) * w + ix;
            const int down  = iclamp(iy+1, 0, h-1) * w + ix;
            const float uAvg = 0.25f * (du0[left] + du0[right] + du0[up] + du0[down]);
            const float vAvg = 0.25f * (dv0[left] + dv0[right] + dv0[up] + dv0[down]);
            const float ix_ = Ix[pos], iy_ = Iy[pos], iz_ = Iz[pos];
            const float denom = alpha * alpha + ix_*ix_ + iy_*iy_;
            const float num   = ix_ * uAvg + iy_ * vAvg + iz_;
            du1[pos] = uAvg - ix_ * num / denom;
            dv1[pos] = vAvg - iy_ * num / denom;
        }
    }
}

// ============================================================
// Image warping: out[pos] = src[pos + (u,v)]
// ============================================================
static void warpImage(const float* src, int w, int h, int sz,
                      const float* u, const float* v, float* out) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
        map(alloc: src[0:sz], u[0:sz], v[0:sz], out[0:sz])
    for (int iy = 0; iy < h; iy++) {
        for (int ix = 0; ix < w; ix++) {
            const int pos = iy * w + ix;
            float sx = (float)ix + u[pos];
            float sy = (float)iy + v[pos];
            sx = sx < 0.f ? 0.f : (sx > (float)(w-1) ? (float)(w-1) : sx);
            sy = sy < 0.f ? 0.f : (sy > (float)(h-1) ? (float)(h-1) : sy);
            const int x0 = (int)sx, y0 = (int)sy;
            const int x1 = iclamp(x0+1, 0, w-1);
            const int y1 = iclamp(y0+1, 0, h-1);
            const float fx = sx - (float)x0, fy = sy - (float)y0;
            out[pos] = (1.f-fy)*((1.f-fx)*src[y0*w+x0] + fx*src[y0*w+x1])
                     +      fy *((1.f-fx)*src[y1*w+x0] + fx*src[y1*w+x1]);
        }
    }
}

// ============================================================
// Downscale by factor 2 (average-pool 4 samples)
// ============================================================
static void downscaleImage(const float* fine, int wF, int hF, int szF,
                            float* coarse, int wC, int hC, int szC) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
        map(alloc: fine[0:szF], coarse[0:szC])
    for (int iy = 0; iy < hC; iy++) {
        for (int ix = 0; ix < wC; ix++) {
            float x = (ix + 0.5f) * 2.f - 0.5f;
            float y = (iy + 0.5f) * 2.f - 0.5f;
            float wx[4] = {x-0.25f, x+0.25f, x,       x      };
            float wy[4] = {y,       y,        y-0.25f, y+0.25f};
            float val = 0.f;
            for (int k = 0; k < 4; k++) {
                int cx = iclamp((int)wx[k], 0, wF-1);
                int cy = iclamp((int)wy[k], 0, hF-1);
                val += fine[cy * wF + cx];
            }
            coarse[iy * wC + ix] = 0.25f * val;
        }
    }
}

// ============================================================
// Upscale flow by factor 2 (bilinear + multiply by scale)
// ============================================================
static void upscaleFlow(const float* coarse, int wC, int hC, int szC,
                         float* fine, int wF, int hF, int szF, float scale) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
        map(alloc: coarse[0:szC], fine[0:szF])
    for (int iy = 0; iy < hF; iy++) {
        for (int ix = 0; ix < wF; ix++) {
            float cx = (ix + 0.5f) / scale - 0.5f;
            float cy = (iy + 0.5f) / scale - 0.5f;
            const int x0 = (int)cx, y0 = (int)cy;
            const float fx = cx - (float)x0, fy = cy - (float)y0;
            const int x1 = iclamp(x0+1, 0, wC-1), y1 = iclamp(y0+1, 0, hC-1);
            const int X0 = iclamp(x0, 0, wC-1),  Y0 = iclamp(y0, 0, hC-1);
            const float v = (1.f-fy)*((1.f-fx)*coarse[Y0*wC+X0] + fx*coarse[Y0*wC+x1])
                          +      fy *((1.f-fx)*coarse[y1*wC+X0] + fx*coarse[y1*wC+x1]);
            fine[iy * wF + ix] = v * scale;
        }
    }
}

// ============================================================
// Coarse-to-fine optical flow computation
// ============================================================
static void computeFlow(float* I0_host, float* I1_host,
                         int width, int height,
                         float alpha, int nLevels, int nWarpIters, int nSolverIters,
                         float* u_host, float* v_host) {
    // Build pyramid level dimensions
    int pw[N_LEVELS], ph[N_LEVELS], psz[N_LEVELS];
    pw[0] = width; ph[0] = height; psz[0] = width * height;
    for (int l = 1; l < nLevels; l++) {
        pw[l] = (pw[l-1]+1)/2;
        ph[l] = (ph[l-1]+1)/2;
        psz[l] = pw[l] * ph[l];
    }

    // Allocate pyramid images on device
    float* dI0[N_LEVELS], *dI1[N_LEVELS];
    for (int l = 0; l < nLevels; l++) {
        dI0[l] = devAlloc(psz[l]);
        dI1[l] = devAlloc(psz[l]);
    }

    // Copy input images to device finest level
    memcpy(/* no-op — host ptrs already set */I0_host, I0_host, 0);
    #pragma omp target enter data map(to: I0_host[0:psz[0]], I1_host[0:psz[0]])
    devCopy(I0_host, dI0[0], psz[0]);
    devCopy(I1_host, dI1[0], psz[0]);
    #pragma omp target exit data map(delete: I0_host[0:psz[0]], I1_host[0:psz[0]])

    // Build pyramid by downscaling
    for (int l = 1; l < nLevels; l++) {
        downscaleImage(dI0[l-1], pw[l-1], ph[l-1], psz[l-1],
                       dI0[l],   pw[l],   ph[l],   psz[l]);
        downscaleImage(dI1[l-1], pw[l-1], ph[l-1], psz[l-1],
                       dI1[l],   pw[l],   ph[l],   psz[l]);
    }

    // Working flow starts at coarsest level
    // u_cur/v_cur hold the current-level flow
    float* u_cur = devAlloc(psz[nLevels-1]);
    float* v_cur = devAlloc(psz[nLevels-1]);
    int    cur_sz = psz[nLevels-1];
    devZero(u_cur, cur_sz);
    devZero(v_cur, cur_sz);

    // Process levels from coarsest to finest
    for (int l = nLevels - 1; l >= 0; l--) {
        const int lsz = psz[l];

        // Upscale flow from coarser level if not at coarsest
        float* ul, *vl;
        if (l == nLevels - 1) {
            // Already initialized to zero
            ul = u_cur; vl = v_cur;
        } else {
            ul = devAlloc(lsz);
            vl = devAlloc(lsz);
            upscaleFlow(u_cur, pw[l+1], ph[l+1], psz[l+1],
                        ul,    pw[l],   ph[l],   lsz, 2.f);
            upscaleFlow(v_cur, pw[l+1], ph[l+1], psz[l+1],
                        vl,    pw[l],   ph[l],   lsz, 2.f);
            devFree(u_cur, cur_sz);
            devFree(v_cur, cur_sz);
            u_cur = ul; v_cur = vl; cur_sz = lsz;
        }

        // Per-level temporaries
        float* Ix   = devAlloc(lsz);
        float* Iy   = devAlloc(lsz);
        float* Iz   = devAlloc(lsz);
        float* I1w  = devAlloc(lsz);
        float* du0  = devAlloc(lsz);
        float* dv0  = devAlloc(lsz);
        float* du1  = devAlloc(lsz);
        float* dv1  = devAlloc(lsz);

        for (int warp = 0; warp < nWarpIters; warp++) {
            // Warp I1 by current flow
            warpImage(dI1[l], pw[l], ph[l], lsz, u_cur, v_cur, I1w);

            // Compute image derivatives
            computeDerivatives(dI0[l], I1w, pw[l], ph[l], lsz, Ix, Iy, Iz);

            // Jacobi iterations for incremental flow
            devZero(du0, lsz);
            devZero(dv0, lsz);

            for (int iter = 0; iter < nSolverIters; iter++) {
                jacobiIteration(du0, dv0, Ix, Iy, Iz,
                                 pw[l], ph[l], lsz, alpha, du1, dv1);
                // Swap ping-pong buffers
                float* tmp = du0; du0 = du1; du1 = tmp;
                tmp = dv0; dv0 = dv1; dv1 = tmp;
            }

            // Update total flow with incremental flow
            devAdd(u_cur, du0, lsz);
            devAdd(v_cur, dv0, lsz);
        }

        devFree(Ix,  lsz); devFree(Iy,  lsz); devFree(Iz,  lsz);
        devFree(I1w, lsz);
        devFree(du0, lsz); devFree(dv0, lsz);
        devFree(du1, lsz); devFree(dv1, lsz);
    }

    // Copy result from device to host
    #pragma omp target update from(u_cur[0:cur_sz])
    #pragma omp target update from(v_cur[0:cur_sz])
    memcpy(u_host, u_cur, cur_sz * sizeof(float));
    memcpy(v_host, v_cur, cur_sz * sizeof(float));

    devFree(u_cur, cur_sz);
    devFree(v_cur, cur_sz);

    for (int l = 0; l < nLevels; l++) {
        devFree(dI0[l], psz[l]);
        devFree(dI1[l], psz[l]);
    }
}

// ============================================================
// Generate synthetic image (checkerboard + sine gradient)
// ============================================================
static void generateImage(float* img, int w, int h, float phase) {
    for (int iy = 0; iy < h; iy++) {
        for (int ix = 0; ix < w; ix++) {
            float xn = ix / (float)w;
            float yn = iy / (float)h;
            float v = 0.5f + 0.3f * sinf(2.f * 3.14159f * (xn + phase))
                           + 0.2f * cosf(4.f * 3.14159f * (yn + phase * 0.5f));
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            img[iy * w + ix] = v;
        }
    }
}

// ============================================================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    printf("Optical flow: %dx%d, alpha=%.2f, levels=%d, warpIters=%d, solverIters=%d\n",
           IMG_W, IMG_H, ALPHA, N_LEVELS, N_WARP_ITERS, N_SOLVER_ITERS);

    float* I0 = (float*)malloc(IMG_SZ * sizeof(float));
    float* I1 = (float*)malloc(IMG_SZ * sizeof(float));
    float* u  = (float*)malloc(IMG_SZ * sizeof(float));
    float* v  = (float*)malloc(IMG_SZ * sizeof(float));

    generateImage(I0, IMG_W, IMG_H, 0.f);
    generateImage(I1, IMG_W, IMG_H, 0.01f);

    // Warm-up
    computeFlow(I0, I1, IMG_W, IMG_H, ALPHA,
                N_LEVELS, N_WARP_ITERS, N_SOLVER_ITERS, u, v);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
        computeFlow(I0, I1, IMG_W, IMG_H, ALPHA,
                    N_LEVELS, N_WARP_ITERS, N_SOLVER_ITERS, u, v);
    }
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
    printf("Average optical flow computation time: %f (us)\n", elapsed_us / repeat);

    // Compute flow checksum
    float sumU = 0.f, sumV = 0.f;
    for (int i = 0; i < IMG_SZ; i++) { sumU += u[i]; sumV += v[i]; }
    printf("Flow checksum: u_sum=%f  v_sum=%f\n", sumU, sumV);

    free(I0); free(I1); free(u); free(v);
    return 0;
}

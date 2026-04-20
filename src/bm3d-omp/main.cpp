// BM3D image denoising benchmark – OpenMP target offloading port
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>

static constexpr int REPEAT       = 100;
static constexpr int IMG_W        = 256;
static constexpr int IMG_H        = 256;
static constexpr int K            = 8;
static constexpr int N_STACK      = 16;
static constexpr float SIGMA      = 25.0f;
static constexpr float L3D        = 2.7f;
static constexpr int PATCH_STRIDE = 8;

static constexpr float C_a    = 1.387039845322148f;
static constexpr float C_b    = 1.306562964876377f;
static constexpr float C_c    = 1.175875602419359f;
static constexpr float C_d    = 0.785694958387102f;
static constexpr float C_e    = 0.541196100146197f;
static constexpr float C_f    = 0.275899379282943f;
static constexpr float C_norm = 0.3535533905932737f;

#pragma omp declare target
void dct8(float* v, int step)
{
    float v0=v[0],       v1=v[step],   v2=v[2*step], v3=v[3*step];
    float v4=v[4*step],  v5=v[5*step], v6=v[6*step], v7=v[7*step];
    float X07P = v0+v7,  X16P = v1+v6,  X25P = v2+v5,  X34P = v3+v4;
    float X07M = v0-v7,  X61M = v6-v1,  X25M = v2-v5,  X43M = v4-v3;
    float X07P34PP = X07P+X34P,  X07P34PM = X07P-X34P;
    float X16P25PP = X16P+X25P,  X16P25PM = X16P-X25P;
    v[0]      = C_norm*(X07P34PP + X16P25PP);
    v[2*step] = C_norm*(C_b*X07P34PM + C_e*X16P25PM);
    v[4*step] = C_norm*(X07P34PP - X16P25PP);
    v[6*step] = C_norm*(C_e*X07P34PM - C_b*X16P25PM);
    v[step]   = C_norm*(C_a*X07M - C_c*X61M + C_d*X25M - C_f*X43M);
    v[3*step] = C_norm*(C_c*X07M + C_f*X61M - C_a*X25M + C_d*X43M);
    v[5*step] = C_norm*(C_d*X07M + C_a*X61M + C_f*X25M - C_c*X43M);
    v[7*step] = C_norm*(C_f*X07M + C_d*X61M + C_c*X25M + C_a*X43M);
}

void idct8(float* v, int step)
{
    float v0=v[0],       v1=v[step],   v2=v[2*step], v3=v[3*step];
    float v4=v[4*step],  v5=v[5*step], v6=v[6*step], v7=v[7*step];
    float Y04P       = v0+v4;
    float Y2b6eP     = C_b*v2 + C_e*v6;
    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f*v7 + C_a*v1 + C_c*v3 + C_d*v5;
    float Y7a1fM3d5cMP = C_a*v7 - C_f*v1 + C_d*v3 - C_c*v5;
    float Y04M       = v0-v4;
    float Y2e6bM     = C_e*v2 - C_b*v6;
    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c*v1 - C_d*v7 - C_f*v3 - C_a*v5;
    float Y1d7cP3a5fMM = C_d*v1 + C_c*v7 - C_a*v3 + C_f*v5;
    v[0]      = C_norm*(Y04P2b6ePP + Y7f1aP3c5dPP);
    v[7*step] = C_norm*(Y04P2b6ePP - Y7f1aP3c5dPP);
    v[4*step] = C_norm*(Y04P2b6ePM + Y7a1fM3d5cMP);
    v[3*step] = C_norm*(Y04P2b6ePM - Y7a1fM3d5cMP);
    v[step]   = C_norm*(Y04M2e6bMP + Y1c7dM3f5aPM);
    v[5*step] = C_norm*(Y04M2e6bMM - Y1d7cP3a5fMM);
    v[2*step] = C_norm*(Y04M2e6bMM + Y1d7cP3a5fMM);
    v[6*step] = C_norm*(Y04M2e6bMP - Y1c7dM3f5aPM);
}

void fwht(float* data, int n)
{
    for (int step = 1; step < n; step <<= 1) {
        for (int i = 0; i < n; i += step * 2) {
            for (int j = i; j < i + step; ++j) {
                float a = data[j], b = data[j + step];
                data[j]        = a + b;
                data[j + step] = a - b;
            }
        }
    }
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
    const int num_ref_x = IMG_W / PATCH_STRIDE;
    const int num_ref_y = IMG_H / PATCH_STRIDE;
    const int num_ref   = num_ref_x * num_ref_y;
    const int total_patches = num_ref * N_STACK;

    int*   h_dx = (int*)malloc(N_STACK * sizeof(int));
    int*   h_dy = (int*)malloc(N_STACK * sizeof(int));
    {
        int ii = 0;
        for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx, ++ii) {
                h_dx[ii] = (dx - 2) * PATCH_STRIDE;
                h_dy[ii] = (dy - 2) * PATCH_STRIDE;
            }
    }

    float* image_d     = (float*)malloc(IMG_H * IMG_W * sizeof(float));
    float* stacks_d    = (float*)malloc((size_t)total_patches * K * K * sizeof(float));
    int*   nonzero_d   = (int*)  malloc(num_ref * sizeof(int));
    float* weights_d   = (float*)malloc(num_ref * sizeof(float));
    float* numerator_d = (float*)malloc(IMG_H * IMG_W * sizeof(float));
    float* denominator_d=(float*)malloc(IMG_H * IMG_W * sizeof(float));
    float* result_d    = (float*)malloc(IMG_H * IMG_W * sizeof(float));
    int*   d_dx        = (int*)  malloc(N_STACK * sizeof(int));
    int*   d_dy        = (int*)  malloc(N_STACK * sizeof(int));

    // Synthetic noisy image
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);
        for (int i = 0; i < IMG_H * IMG_W; ++i)
            image_d[i] = dist(rng);
    }
    memcpy(d_dx, h_dx, N_STACK * sizeof(int));
    memcpy(d_dy, h_dy, N_STACK * sizeof(int));

    const int img_size = IMG_H * IMG_W;
    const int stacks_size = total_patches * K * K;

    #pragma omp target enter data map(alloc: image_d[0:img_size], stacks_d[0:stacks_size], \
        nonzero_d[0:num_ref], weights_d[0:num_ref], numerator_d[0:img_size], \
        denominator_d[0:img_size], result_d[0:img_size], d_dx[0:N_STACK], d_dy[0:N_STACK])

    #pragma omp target update to(image_d[0:img_size], d_dx[0:N_STACK], d_dy[0:N_STACK])

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < REPEAT; ++rep) {
        // Reset aggregation buffers on device
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < img_size; i++) { numerator_d[i] = 0.0f; denominator_d[i] = 0.0f; }
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < num_ref; i++) nonzero_d[i] = 0;

        // Step 1: Gather patches
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int idx = 0; idx < total_patches; idx++) {
            const int ref = idx / N_STACK;
            const int s   = idx % N_STACK;
            const int rx  = (ref % num_ref_x) * PATCH_STRIDE;
            const int ry  = (ref / num_ref_x) * PATCH_STRIDE;
            int px = rx + d_dx[s];
            int py = ry + d_dy[s];
            if (px < 0) px = 0; if (px > IMG_W - K) px = IMG_W - K;
            if (py < 0) py = 0; if (py > IMG_H - K) py = IMG_H - K;
            const int base = idx * K * K;
            for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj)
                    stacks_d[base + ki*K + kj] = image_d[(py + ki)*IMG_W + (px + kj)];
        }

        // Step 2: 2D DCT on every patch
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int idx = 0; idx < total_patches; idx++) {
            const int base = idx * K * K;
            float blk[K * K];
            for (int i = 0; i < K*K; ++i) blk[i] = stacks_d[base + i];
            for (int r = 0; r < K; ++r) dct8(blk + r*K, 1);
            for (int c = 0; c < K; ++c) dct8(blk + c,   K);
            for (int i = 0; i < K*K; ++i) stacks_d[base + i] = blk[i];
        }

        // Step 3: FWHT + threshold + IFWHT + weight
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int idx = 0; idx < num_ref * K * K; idx++) {
            const int ref = idx / (K * K);
            const int kij = idx % (K * K);
            float z[N_STACK];
            for (int s = 0; s < N_STACK; ++s)
                z[s] = stacks_d[(ref * N_STACK + s) * K*K + kij];
            fwht(z, N_STACK);
            const float thr = L3D * sqrtf((float)N_STACK) * SIGMA;
            int nz = 0;
            for (int s = 0; s < N_STACK; ++s) {
                if (fabsf(z[s]) < thr) z[s] = 0.0f;
                else ++nz;
            }
            fwht(z, N_STACK);
            const float inv_n = 1.0f / N_STACK;
            for (int s = 0; s < N_STACK; ++s)
                stacks_d[(ref * N_STACK + s) * K*K + kij] = z[s] * inv_n;
            #pragma omp atomic update
            nonzero_d[ref] += nz;
        }

        // Step 3b: weights
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int ref = 0; ref < num_ref; ref++) {
            const int nz = nonzero_d[ref];
            weights_d[ref] = (nz > 0) ? 1.0f / (float)nz : 1.0f;
        }

        // Step 4: 2D IDCT
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int idx = 0; idx < total_patches; idx++) {
            const int base = idx * K * K;
            float blk[K * K];
            for (int i = 0; i < K*K; ++i) blk[i] = stacks_d[base + i];
            for (int r = 0; r < K; ++r) idct8(blk + r*K, 1);
            for (int c = 0; c < K; ++c) idct8(blk + c,   K);
            for (int i = 0; i < K*K; ++i) stacks_d[base + i] = blk[i];
        }

        // Step 5: Aggregate
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int idx = 0; idx < total_patches * K * K; idx++) {
            const int ref = idx / (N_STACK * K * K);
            const int rem = idx % (N_STACK * K * K);
            const int s   = rem / (K * K);
            const int kij = rem % (K * K);
            const int ki  = kij / K;
            const int kj  = kij % K;
            const int rx = (ref % num_ref_x) * PATCH_STRIDE;
            const int ry = (ref / num_ref_x) * PATCH_STRIDE;
            int px = rx + d_dx[s];
            int py = ry + d_dy[s];
            if (px < 0) px = 0; if (px > IMG_W - K) px = IMG_W - K;
            if (py < 0) py = 0; if (py > IMG_H - K) py = IMG_H - K;
            const float wp  = weights_d[ref];
            const float val = stacks_d[(ref * N_STACK + s) * K*K + kij];
            const int   img_idx = (py + ki)*IMG_W + (px + kj);
            #pragma omp atomic update
            numerator_d[img_idx] += val * wp;
            #pragma omp atomic update
            denominator_d[img_idx] += wp;
        }

        // Step 6: Normalise
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < img_size; i++) {
            const float d = denominator_d[i];
            result_d[i] = (d > 0.0f) ? numerator_d[i] / d : image_d[i];
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Verify: mean output pixel value
    double sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:sum) thread_limit(256)
    for (int i = 0; i < img_size; i++) sum += result_d[i];

    printf("BM3D OpenMP benchmark\n");
    printf("Image %dx%d, patch %dx%d, stack %d, iterations %d\n",
           IMG_W, IMG_H, K, K, N_STACK, REPEAT);
    printf("Total time   : %.3f s\n", elapsed);
    printf("Avg per iter : %.3f ms\n", elapsed * 1e3 / REPEAT);
    printf("Result mean  : %.3f\n", sum / img_size);

    #pragma omp target exit data map(delete: image_d[0:img_size], stacks_d[0:stacks_size], \
        nonzero_d[0:num_ref], weights_d[0:num_ref], numerator_d[0:img_size], \
        denominator_d[0:img_size], result_d[0:img_size], d_dx[0:N_STACK], d_dy[0:N_STACK])

    free(image_d); free(stacks_d); free(nonzero_d); free(weights_d);
    free(numerator_d); free(denominator_d); free(result_d);
    free(d_dx); free(d_dy); free(h_dx); free(h_dy);
    return 0;
}

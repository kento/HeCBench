// MNIST CNN - OpenMP target offloading port
// Data files required in ./data/:
//   train-images.idx3-ubyte, train-labels.idx1-ubyte
//   t10k-images.idx3-ubyte,  t10k-labels.idx1-ubyte
//
// Network architecture (same as CUDA version):
//   l_input : 28x28 input
//   l_c1    : conv 5x5, 6 features  -> 6x24x24
//   l_s1    : subsample 4x4         -> 6x6x6
//   l_f     : fully-connected       -> 10 outputs

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

static const float dt        = 1.0e-1f;
static const float threshold = 1.0e-2f;

// -------------------------------------------------------------------------
// sigmoid: available on both host and device
// -------------------------------------------------------------------------
#pragma omp declare target
static float step_function(float v) {
    return 1.f / (1.f + expf(-v));
}
#pragma omp end declare target

// -------------------------------------------------------------------------
// Layer descriptor
// -------------------------------------------------------------------------
struct Layer {
    int M, N, O;
    float *output, *preact, *bias, *weight;
    float *d_output, *d_preact, *d_weight;

    Layer(int M_, int N_, int O_) : M(M_), N(N_), O(O_) {
        output   = (float*)malloc(O * sizeof(float));
        preact   = (float*)malloc(O * sizeof(float));
        d_output = (float*)malloc(O * sizeof(float));
        d_preact = (float*)malloc(O * sizeof(float));
        // Allocate at least 1 element so pointers are never null
        int bias_sz = N > 0 ? N : 1;
        int wt_sz   = M*N > 0 ? M*N : 1;
        bias     = (float*)malloc(bias_sz * sizeof(float));
        weight   = (float*)malloc(wt_sz   * sizeof(float));
        d_weight = (float*)malloc(wt_sz   * sizeof(float));

        // Local pointer copies required: member vars not valid in OMP map clauses
        {
            float* _out = output; float* _pre = preact;
            float* _dout = d_output; float* _dpre = d_preact;
            int sz = O;
            #pragma omp target enter data map(alloc: _out[0:sz], _pre[0:sz], _dout[0:sz], _dpre[0:sz])
        }
        if (N > 0) {
            float* _bias = bias; int n = N;
            #pragma omp target enter data map(alloc: _bias[0:n])
        }
        if (M*N > 0) {
            float* _wt = weight; float* _dwt = d_weight; int mn = M*N;
            #pragma omp target enter data map(alloc: _wt[0:mn], _dwt[0:mn])
        }

        // Random initialisation of bias and weight
        for (int i = 0; i < N; i++) {
            bias[i] = 0.5f - (float)rand() / RAND_MAX;
            for (int j = 0; j < M; j++)
                weight[i*M + j] = 0.5f - (float)rand() / RAND_MAX;
        }
        if (N > 0) {
            float* _bias = bias; int n = N;
            #pragma omp target update to(_bias[0:n])
        }
        if (M*N > 0) {
            float* _wt = weight; int mn = M*N;
            #pragma omp target update to(_wt[0:mn])
        }

        clear();
        bp_clear();
    }

    ~Layer() {
        if (N > 0) {
            float* _bias = bias; int n = N;
            #pragma omp target exit data map(delete: _bias[0:n])
        }
        if (M*N > 0) {
            float* _wt = weight; float* _dwt = d_weight; int mn = M*N;
            #pragma omp target exit data map(delete: _wt[0:mn], _dwt[0:mn])
        }
        {
            float* _out = output; float* _pre = preact;
            float* _dout = d_output; float* _dpre = d_preact;
            int sz = O;
            #pragma omp target exit data map(delete: _out[0:sz], _pre[0:sz], _dout[0:sz], _dpre[0:sz])
        }
        free(output); free(preact); free(bias); free(weight);
        free(d_output); free(d_preact); free(d_weight);
    }

    void setOutput(float* data) {
        memcpy(output, data, O * sizeof(float));
        float* _out = output; int sz = O;
        #pragma omp target update to(_out[0:sz])
    }

    void clear() {
        float* out = output;
        float* pre = preact;
        int sz = O;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < sz; i++) {
            out[i] = 0.0f;
            pre[i] = 0.0f;
        }
    }

    void bp_clear() {
        float* dout = d_output;
        float* dpre = d_preact;
        int sz_O = O;
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int i = 0; i < sz_O; i++) {
            dout[i] = 0.0f;
            dpre[i] = 0.0f;
        }
        if (M*N > 0) {
            float* dwt = d_weight;
            int sz_MN = M*N;
            #pragma omp target teams distribute parallel for thread_limit(256)
            for (int i = 0; i < sz_MN; i++)
                dwt[i] = 0.0f;
        }
    }
};

// -------------------------------------------------------------------------
// Forward pass kernels
// -------------------------------------------------------------------------

static void fp_preact_c1(Layer &input, Layer &c1)
{
    const int N = 5*5*6*24*24;
    float* w   = c1.weight;
    float* pa  = c1.preact;
    float* inp = input.output;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int kx = idx % 5;  idx /= 5;
        int ky = idx % 5;  idx /= 5;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 24; idx /= 24;
        int ox = idx % 24;
        #pragma omp atomic update
        pa[f*24*24 + oy*24 + ox] += w[f*5*5 + ky*5 + kx] * inp[(oy+ky)*28 + (ox+kx)];
    }
}

static void fp_bias_c1(Layer &c1)
{
    const int N = 6*24*24;
    float* pa   = c1.preact;
    float* bias = c1.bias;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int f = n / (24*24);
        pa[n] += bias[f];
    }
}

static void apply_step_function(Layer &l)
{
    float* pre = l.preact;
    float* out = l.output;
    int sz = l.O;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < sz; i++)
        out[i] = step_function(pre[i]);
}

static void fp_preact_s1(Layer &c1, Layer &s1)
{
    const int N = 4*4*6*6*6;
    float* w  = s1.weight;
    float* pa = s1.preact;
    float* in = c1.output;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int kx = idx % 4;  idx /= 4;
        int ky = idx % 4;  idx /= 4;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 6;  idx /= 6;
        int ox = idx % 6;
        #pragma omp atomic update
        pa[f*6*6 + oy*6 + ox] += w[ky*4 + kx] * in[f*24*24 + (oy*4+ky)*24 + (ox*4+kx)];
    }
}

static void fp_bias_s1(Layer &s1)
{
    const int N = 6*6*6;
    float* pa   = s1.preact;
    float* bias = s1.bias;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++)
        pa[n] += bias[0];
}

static void fp_preact_f(Layer &s1, Layer &lf)
{
    const int N = 10*6*6*6;
    float* w  = lf.weight;
    float* pa = lf.preact;
    float* in = s1.output;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int o = idx % 10; idx /= 10;
        int f = idx % 6;  idx /= 6;
        int y = idx % 6;  idx /= 6;
        int x = idx % 6;
        #pragma omp atomic update
        pa[o] += w[o*6*6*6 + f*6*6 + y*6 + x] * in[f*6*6 + y*6 + x];
    }
}

static void fp_bias_f(Layer &lf)
{
    float* pa   = lf.preact;
    float* bias = lf.bias;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < 10; i++)
        pa[i] += bias[i];
}

// -------------------------------------------------------------------------
// Backward pass kernels
// -------------------------------------------------------------------------

static void makeError(Layer &lf, unsigned int Y)
{
    float* dp  = lf.d_preact;
    float* out = lf.output;
    unsigned int y = Y;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < 10; i++)
        dp[i] = ((int)y == i ? 1.0f : 0.0f) - out[i];
}

static void bp_weight_f(Layer &lf, Layer &s1)
{
    const int N = 10*6*6*6;
    float* dw = lf.d_weight;
    float* dp = lf.d_preact;
    float* in = s1.output;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int o = idx % 10; idx /= 10;
        int f = idx % 6;  idx /= 6;
        int y = idx % 6;  idx /= 6;
        int x = idx % 6;
        dw[o*6*6*6 + f*6*6 + y*6 + x] = dp[o] * in[f*6*6 + y*6 + x];
    }
}

static void bp_bias_f(Layer &lf)
{
    float* bias = lf.bias;
    float* dp   = lf.d_preact;
    float _dt   = dt;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < 10; i++)
        bias[i] += _dt * dp[i];
}

static void bp_output_s1(Layer &s1, Layer &lf)
{
    const int N = 10*6*6*6;
    float* dout = s1.d_output;
    float* wf   = lf.weight;
    float* dpf  = lf.d_preact;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int o = idx % 10; idx /= 10;
        int f = idx % 6;  idx /= 6;
        int y = idx % 6;  idx /= 6;
        int x = idx % 6;
        #pragma omp atomic update
        dout[f*6*6 + y*6 + x] += wf[o*6*6*6 + f*6*6 + y*6 + x] * dpf[o];
    }
}

static void bp_preact_s1(Layer &s1)
{
    const int N = 6*6*6;
    float* dp   = s1.d_preact;
    float* dout = s1.d_output;
    float* pre  = s1.preact;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        float o = step_function(pre[n]);
        dp[n] = dout[n] * o * (1.f - o);
    }
}

static void bp_weight_s1(Layer &s1, Layer &c1)
{
    const int N = 4*4*6*6*6;
    float* dw  = s1.d_weight;
    float* dp  = s1.d_preact;
    float* pc1 = c1.output;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int kx = idx % 4;  idx /= 4;
        int ky = idx % 4;  idx /= 4;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 6;  idx /= 6;
        int ox = idx % 6;
        #pragma omp atomic update
        dw[ky*4 + kx] += dp[f*6*6 + oy*6 + ox] * pc1[f*24*24 + (oy*4+ky)*24 + (ox*4+kx)];
    }
}

static void bp_bias_s1(Layer &s1)
{
    const int N = 6*6*6;
    const float d = 216.f;
    float* bias = s1.bias;
    float* dp   = s1.d_preact;
    float _dt   = dt;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        #pragma omp atomic update
        bias[0] += _dt * dp[n] / d;
    }
}

static void bp_output_c1(Layer &c1, Layer &s1)
{
    const int N = 4*4*6*6*6;
    float* dout = c1.d_output;
    float* ws1  = s1.weight;
    float* dps1 = s1.d_preact;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int kx = idx % 4;  idx /= 4;
        int ky = idx % 4;  idx /= 4;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 6;  idx /= 6;
        int ox = idx % 6;
        #pragma omp atomic update
        dout[f*24*24 + (oy*4+ky)*24 + (ox*4+kx)] += ws1[ky*4 + kx] * dps1[f*6*6 + oy*6 + ox];
    }
}

static void bp_preact_c1(Layer &c1)
{
    const int N = 6*24*24;
    float* dp   = c1.d_preact;
    float* dout = c1.d_output;
    float* pre  = c1.preact;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        float o = step_function(pre[n]);
        dp[n] = dout[n] * o * (1.f - o);
    }
}

static void bp_weight_c1(Layer &c1, Layer &input)
{
    const int N = 6*5*5*24*24;
    const float d = 576.f;
    float* dw  = c1.d_weight;
    float* dp  = c1.d_preact;
    float* inp = input.output;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int idx = n;
        int f  = idx % 6;  idx /= 6;
        int kx = idx % 5;  idx /= 5;
        int ky = idx % 5;  idx /= 5;
        int oy = idx % 24; idx /= 24;
        int ox = idx % 24;
        #pragma omp atomic update
        dw[f*5*5 + ky*5 + kx] += dp[f*24*24 + oy*24 + ox] * inp[(oy+ky)*28 + (ox+kx)] / d;
    }
}

static void bp_bias_c1(Layer &c1)
{
    const int N = 6*24*24;
    const float d = 576.f;
    float* bias = c1.bias;
    float* dp   = c1.d_preact;
    float _dt   = dt;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int n = 0; n < N; n++) {
        int f = n / (24*24);
        #pragma omp atomic update
        bias[f] += _dt * dp[n] / d;
    }
}

static void apply_grad(float* wt, float* dw, int sz)
{
    float _dt = dt;
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < sz; i++)
        wt[i] += _dt * dw[i];
}

// -------------------------------------------------------------------------
// Compute L2 norm of d_preact on host (replaces cublas snrm2)
// -------------------------------------------------------------------------
static float compute_norm(float* v, int n)
{
    #pragma omp target update from(v[0:n])
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum);
}

// -------------------------------------------------------------------------
// Forward and backward pass wrappers
// -------------------------------------------------------------------------
static void forward_pass(double data[28][28],
                         Layer &l_input, Layer &l_c1, Layer &l_s1, Layer &l_f)
{
    l_input.clear(); l_c1.clear(); l_s1.clear(); l_f.clear();

    float inp[28*28];
    for (int i = 0; i < 28; i++)
        for (int j = 0; j < 28; j++)
            inp[i*28+j] = (float)data[i][j];
    l_input.setOutput(inp);

    fp_preact_c1(l_input, l_c1);
    fp_bias_c1(l_c1);
    apply_step_function(l_c1);

    fp_preact_s1(l_c1, l_s1);
    fp_bias_s1(l_s1);
    apply_step_function(l_s1);

    fp_preact_f(l_s1, l_f);
    fp_bias_f(l_f);
    apply_step_function(l_f);
}

static void back_pass(Layer &l_input, Layer &l_c1, Layer &l_s1, Layer &l_f)
{
    bp_weight_f(l_f, l_s1);
    bp_bias_f(l_f);

    bp_output_s1(l_s1, l_f);
    bp_preact_s1(l_s1);
    bp_weight_s1(l_s1, l_c1);
    bp_bias_s1(l_s1);

    bp_output_c1(l_c1, l_s1);
    bp_preact_c1(l_c1);
    bp_weight_c1(l_c1, l_input);
    bp_bias_c1(l_c1);

    apply_grad(l_f.weight,  l_f.d_weight,  l_f.M  * l_f.N);
    apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
    apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    if (argc != 2) { printf("Usage: %s <iterations>\n", argv[0]); return 1; }
    const int iter = atoi(argv[1]);

    srand(123);

    // Load MNIST
    mnist_data *train_set = nullptr, *test_set = nullptr;
    unsigned int train_cnt = 0, test_cnt = 0;
    if (mnist_load("data/train-images.idx3-ubyte",
                   "data/train-labels.idx1-ubyte",
                   &train_set, &train_cnt) ||
        mnist_load("data/t10k-images.idx3-ubyte",
                   "data/t10k-labels.idx1-ubyte",
                   &test_set, &test_cnt)) {
        fprintf(stderr, "Failed to load MNIST data from ./data/\n");
        return 1;
    }

    // Build network layers (same sizes as CUDA version)
    Layer l_input(0,     0,  28*28);
    Layer l_c1   (5*5,   6,  24*24*6);
    Layer l_s1   (4*4,   1,  6*6*6);
    Layer l_f    (6*6*6, 10, 10);

    auto t1 = std::chrono::high_resolution_clock::now();

    // ---- Training ----
    fprintf(stdout, "Learning\n");
    int remaining = iter;
    while (remaining < 0 || remaining-- > 0) {
        float err = 0.f;
        for (unsigned i = 0; i < train_cnt; i++) {
            forward_pass(train_set[i].data, l_input, l_c1, l_s1, l_f);

            l_f.bp_clear();
            l_s1.bp_clear();
            l_c1.bp_clear();

            makeError(l_f, train_set[i].label);
            err += compute_norm(l_f.d_preact, 10);

            back_pass(l_input, l_c1, l_s1, l_f);
        }
        err /= train_cnt;
        fprintf(stdout, "error: %e\n", err);
        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }
    }

    // ---- Testing ----
    fprintf(stdout, "Testing\n");
    int errors = 0;
    for (unsigned i = 0; i < test_cnt; i++) {
        forward_pass(test_set[i].data, l_input, l_c1, l_s1, l_f);

        { float* _fo = l_f.output; int fsz = 10;
          #pragma omp target update from(_fo[0:fsz])
        }
        unsigned int pred = 0;
        for (int k = 1; k < 10; k++)
            if (l_f.output[k] > l_f.output[pred]) pred = k;
        if (pred != test_set[i].label) errors++;
    }
    fprintf(stdout, "Error Rate: %.2f%%\n", 100.0 * errors / (double)test_cnt);

    auto t2 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("Total time (learn + test) %.6f secs\n", total / 1e6);

    return 0;
}

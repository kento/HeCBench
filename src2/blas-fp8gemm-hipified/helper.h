/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>

#include <hipblaslt.h>
#include <cuda_fp8.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

inline void checkCudaStatus(hipError_t status) {
    if (status != hipSuccess) {
        printf("cuda API failed with status %d: %s\n", status, hipGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template <typename InType, typename CType, typename OutType = InType, typename ComputeType = OutType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k,
            ComputeType alpha = ComputeType{0.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
            ComputeType Ascale = ComputeType{2.0f}, ComputeType Bscale = ComputeType{0.5f},
            ComputeType Cscale = ComputeType{1.0f}, ComputeType Dscale = ComputeType{1.0f}) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), 
        Ahost(m * k * N), Bhost(n * k * N), Chost(m * n * N), Dhost(m * n * N),
        biasHost(m * N), AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale) {
        checkCublasStatus(hipblasLtCreate(&ltHandle));
        checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(CType)));
        checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&Ddev), m * n * N  * sizeof(OutType)));
        checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&biasDev), m * N * sizeof(OutType)));
        checkCudaStatus(hipMalloc(&workspace, workspaceSize));
        checkCudaStatus(hipStreamCreate(&stream));

        // Currently only fp8 supports per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value || std::is_same<InType, __nv_fp8_e5m2>::value;

        if (perTensorScalingEnabled) {
            checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            checkCudaStatus(hipMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

        fillData();
    }

    ~TestBench() {
        checkCublasStatus(hipblasLtDestroy(ltHandle));
        checkCudaStatus(hipFree(Adev));
        checkCudaStatus(hipFree(Bdev));
        checkCudaStatus(hipFree(Cdev));
        checkCudaStatus(hipFree(Ddev));
        checkCudaStatus(hipFree(biasDev));
        checkCudaStatus(hipFree(workspace));
        if (perTensorScalingEnabled) {
            checkCudaStatus(hipFree(AscaleDev));
            checkCudaStatus(hipFree(BscaleDev));
            checkCudaStatus(hipFree(CscaleDev));
            checkCudaStatus(hipFree(DscaleDev));
            checkCudaStatus(hipFree(DamaxDev));
        }
        checkCudaStatus(hipStreamDestroy(stream));
    }

    void fillData() {
        for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(i);
        for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(i);
        for (int i = 0; i < m * n * N; i++) Chost[i] = CType(-i);
        for (int i = 0; i < m * N; i++) biasHost[i] = InType(i + 1);
    }

    void copyDataToDevice() {
        checkCudaStatus(hipMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), hipMemcpyHostToDevice, stream));
        checkCudaStatus(hipMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), hipMemcpyHostToDevice, stream));
        checkCudaStatus(hipMemcpyAsync(Cdev, Chost.data(), Chost.size() * sizeof(Chost[0]), hipMemcpyHostToDevice, stream));
        checkCudaStatus(hipMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), hipMemcpyHostToDevice));
        if (perTensorScalingEnabled) {
            checkCudaStatus(hipMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), hipMemcpyHostToDevice));
            checkCudaStatus(hipMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), hipMemcpyHostToDevice));
            checkCudaStatus(hipMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), hipMemcpyHostToDevice));
            checkCudaStatus(hipMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), hipMemcpyHostToDevice));
            checkCudaStatus(hipMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), hipMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        checkCudaStatus(hipMemcpyAsync(Dhost.data(), Ddev, Dhost.size() * sizeof(Dhost[0]), hipMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(hipStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool perTensorScalingEnabled;
    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<CType> Chost;
    std::vector<OutType> Dhost, biasHost;
    void *workspace;
    InType *Adev, *Bdev;
    CType *Cdev;
    OutType *Ddev, *biasDev;
    hipStream_t stream;
    hipblasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
};

template <>
inline void TestBench<__half, __half,  __half, float>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Chost[i] = __float2half_rn(-i);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, __half, hipComplex>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Chost[i] = __float2half_rn(-i/100.);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

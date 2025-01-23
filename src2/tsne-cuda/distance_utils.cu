/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief Implementation of different distances
 *
 * @file distance_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include <chrono>
#include <future>
#include <fstream>
#include "distance_utils.h"

void tsne::utils::KNearestNeighbors(
    std::string data_folder,
    int64_t* indices,               // output indices
    float* distances,               // output distances
    const int num_dims,             // number of pixels per image = 784
    const int num_points,           // number of images
    const int num_near_neighbors)   // default value is 32
{
    std::cout << "num_points: " << num_points << std::endl;
    std::cout << "num_dims: " << num_dims << std::endl;
    std::cout << "num_near_neighbors: " << num_near_neighbors << std::endl;

    std::string distances_file = data_folder + "distances";
    std::ifstream dist_file(distances_file, std::ios::in | std::ios::binary);
    float dist_val = 0.0f;
    for (int i = 0 ; i < num_points*num_near_neighbors; ++i) {
        dist_file.read(reinterpret_cast<char*>(&dist_val), sizeof(float));
        if (dist_file.gcount() != 4) {
            std::cout << "E: distance file read error." << std::endl;
        }
        distances[i] = dist_val;
    }

    std::string indices_file = data_folder + "indices";
    std::ifstream ind_file(indices_file, std::ios::in | std::ios::binary);
    long ind_val = 0;
    for (int i = 0 ; i < num_points*num_near_neighbors; ++i) {
        ind_file.read(reinterpret_cast<char*>(&ind_val), sizeof(long));
        if (ind_file.gcount() != 8) {
            std::cout << "E: index file read error." << std::endl;
        }
        indices[i] = ind_val;
    }

    // std::cout << "distance values:\n";
    // for (int i = 0; i < 8; ++i) {
    //     std::cout << distances[i] << " ";
    // }
    // std::cout << "...\n";

    // std::cout << "index values:\n";
    // for (int i = 0; i < 8; ++i) {
    //     std::cout << indices[i] << " ";
    // }
    // std::cout << "...\n";
}

// TODO: Add -1 notification here... and how to deal with it if it happens
// TODO: Maybe think about getting FAISS to return integers (long-term todo)
__global__
void PostprocessNeighborIndicesKernel(
    volatile int* __restrict__ pij_indices,
    const long*   __restrict__ knn_indices,
    const int num_points,
    const int num_neighbors)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points * num_neighbors)
        return;
    pij_indices[tid] = (int)knn_indices[tid];
}

void tsne::utils::PostprocessNeighborIndices(
    thrust::device_vector<int>& pij_indices,
    thrust::device_vector<int64_t>& knn_indices,
    const int num_points,
    const int num_neighbors)
{
    const int num_threads = 256;
    const int num_blocks = iDivUp(num_points * num_neighbors, num_threads);
    PostprocessNeighborIndicesKernel<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(pij_indices.data()),
        thrust::raw_pointer_cast(knn_indices.data()),
        num_points,
        num_neighbors
    );
    CUDA_CHECK_LAST_ERROR()
    GpuErrorCheck(cudaDeviceSynchronize());
}

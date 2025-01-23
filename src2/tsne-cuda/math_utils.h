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
 * @brief Utilities for doing math of different sorts
 *
 * @file math_utils.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_MATH_UTILS_H_
#define SRC_INCLUDE_UTIL_MATH_UTILS_H_

#include "common.h"
#include "options.h"
#include "matrix_broadcast_utils.h"
#include "thrust_transform_functions.h"

namespace tsne
{
    namespace utils
    {

        /**
         * @brief Compute the L2 Norm of a device vector
         *
         * @param d_vector The vector to compute the norm of
         * @return float The L2 Norm
         */
        float L2NormDeviceVector(const thrust::device_vector<float>& d_vector);

        /**
         * @brief Max-normalize a device vector in place
         *
         * @param d_vector The vector to normalize
         */
        void MaxNormalizeDeviceVector(thrust::device_vector<float>& d_vector);

        // V2 Symmetrization
        void SymmetrizeMatrixV2(
            thrust::device_vector<float>&   pij_symmetric,
            thrust::device_vector<float>&   pij_nonsymmetric,
            thrust::device_vector<int32_t>& pij_indices,
            const int num_points,
            const int num_neighbors);

    } // namespace utils
} // namespace tsne

#endif // SRC_INCLUDE_UTIL_MATH_UTILS_H_

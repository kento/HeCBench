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

#ifndef NBODYFFT_H
#define NBODYFFT_H

#include <chrono>
#include <complex>
#include <thrust/complex.h>
#include "common.h"
#include "cuda_utils.h"
#include "debug_utils.h"
#include "matrix_broadcast_utils.h"

namespace tsne
{

void PrecomputeFFT2D(
    // cufftHandle& plan_kernel_tilde,
    float x_max,
    float x_min,
    float y_max,
    float y_min,
    int n_boxes,
    int n_interpolation_points,
    thrust::device_vector<float>& box_lower_bounds_device,
    thrust::device_vector<float>& box_upper_bounds_device,
    thrust::device_vector<float>& kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>>& fft_kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>>& fft_scratchpad_device, double& duration); // added

void NbodyFFT2D(
    // cufftHandle& plan_dft,
    // cufftHandle& plan_idft,
    thrust::device_vector<thrust::complex<float>>& fft_kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>>& fft_w_coefficients,
    int   N,
    int   n_terms,
    int   n_boxes,
    int   n_interpolation_points,
    int   n_total_boxes,
    int   total_interpolation_points,
    float coord_min,
    float box_width,
    int   n_fft_coeffs_half,
    int   n_fft_coeffs,
    thrust::device_vector<float>& fft_input,
    thrust::device_vector<float>& fft_output,
    thrust::device_vector<int>&   point_box_idx_device,
    thrust::device_vector<float>& x_in_box_device,
    thrust::device_vector<float>& y_in_box_device,
    thrust::device_vector<float>& points_device,
    thrust::device_vector<float>& box_lower_bounds_device,
    thrust::device_vector<float>& y_tilde_spacings_device,
    thrust::device_vector<float>& denominator_device,
    thrust::device_vector<float>& y_tilde_values,
    // thrust::device_vector<float>& all_interpolated_values_device,
    // thrust::device_vector<float>& output_values,
    // thrust::device_vector<int>&   all_interpolated_indices,
    // thrust::device_vector<int>&   output_indices,
    thrust::device_vector<float>& w_coefficients_device,
    thrust::device_vector<float>& chargesQij_device,
    thrust::device_vector<float>& x_interpolated_values_device,
    thrust::device_vector<float>& y_interpolated_values_device,
    thrust::device_vector<float>& potentialsQij_device,
    thrust::device_vector<thrust::complex<float>>& fft_scratchpad_device, double& duration); // added

} // namespace tsne

#endif

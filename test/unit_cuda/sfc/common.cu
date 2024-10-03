/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  Tests for SFC related GPU device functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/sfc/common.hpp"

using namespace cstone;

template<class T>
__global__ void nzPlaceKernel(T* values)
{
    T laneValue = threadIdx.x;

    values[threadIdx.x] = lastNzPlace(laneValue);
}

template<class T>
void nzPlaceTest()
{
    thrust::host_vector<T> h_v(GpuConfig::warpSize);
    thrust::device_vector<T> d_v = h_v;

    nzPlaceKernel<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));
    h_v = d_v;

    EXPECT_EQ(h_v[0], maxTreeLevel<T>{});
    EXPECT_EQ(h_v[1], maxTreeLevel<T>{});
    EXPECT_EQ(h_v[4], maxTreeLevel<T>{});
    EXPECT_EQ(h_v[8], maxTreeLevel<T>{} - 1);
}

TEST(NzPlaceGpu, lastNzPlace)
{
    nzPlaceTest<uint32_t>();
    nzPlaceTest<uint64_t>();
}
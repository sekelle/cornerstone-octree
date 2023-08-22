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
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <bitset>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/traversal/groups.cuh"
#include "cstone/tree/cs_util.hpp"
#include "cstone/util/util.hpp"

using namespace cstone;

constexpr size_t targetSize = 64;
constexpr size_t nwt        = targetSize / GpuConfig::warpSize;
using SplitType             = util::array<GpuConfig::ThreadMask, nwt>;

TEST(TargetGroups, t0)
{
    using T              = double;
    LocalIndex numGroups = 4;
    LocalIndex groupSize = 8;
    LocalIndex first     = 4;
    LocalIndex last      = 34;

    thrust::device_vector<LocalIndex> groups(numGroups + 1);

    groupTargets<T><<<iceil(last - first, 64), 64>>>(first, last, nullptr, nullptr, nullptr, nullptr, groupSize,
                                                     rawPtr(groups), iceil(last - first, groupSize));

    thrust::host_vector<LocalIndex> hgroups = groups;

    thrust::host_vector<LocalIndex> ref = std::vector<LocalIndex>{4, 12, 20, 28, 34};

    EXPECT_EQ(hgroups, ref);
}

__device__ constexpr util::array<int, 2> laneSeg(int idx, int warpSize_) { return {idx % warpSize, idx / warpSize_}; }

//! @brief test input setup for findSplits
template<size_t N>
__global__ void findSplitTester(util::array<GpuConfig::ThreadMask, N>* splits)
{
    using T          = double;
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    util::array<Vec4<T>, N> pos;
    for (int k = 0; k < N; ++k)
    {
        T x    = T(laneIdx) + k * GpuConfig::warpSize;
        pos[k] = Vec4<T>{x, x, x, T(0)};
    }

    // introduce a split at position 0
    if (laneIdx == 0) { pos[0] = {-1, -1, -1, 0}; }

    // introduce a split at position 31
    if (laneIdx == 31)
    {
        pos[0][0] -= 0.5;
        pos[0][1] -= 0.5;
        pos[0][2] -= 0.5;
    }

    // introduce a split at position 33
    auto lk = laneSeg(33, GpuConfig::warpSize);
    if (lk[0] == laneIdx && lk[1] < N)
    {
        pos[lk[1]][0] -= 0.5;
        pos[lk[1]][1] -= 0.5;
        pos[lk[1]][2] -= 0.5;
    }

    *splits = findSplits(pos, 3.01);
}

TEST(TargetGroups, findSplits)
{
    {
        thrust::device_vector<SplitType> d_splits(1);
        findSplitTester<<<1, GpuConfig::warpSize>>>(rawPtr(d_splits));
        SplitType split = d_splits[0];

        std::bitset<targetSize> splitBits;
        for (int k = nwt; k >= 0; --k)
        {
            splitBits <<= GpuConfig::warpSize;
            splitBits |= split[k];
        }

        EXPECT_EQ(splitBits.count(), 3);
        EXPECT_EQ(splitBits[0], 1);
        EXPECT_EQ(splitBits[31], 1);
        EXPECT_EQ(splitBits[33], 1);
    }
}

__global__ void makeSplitTester(SplitType splitMask, LocalIndex* splitLengths) { makeSplits(splitMask, splitLengths); }

TEST(TargetGroups, makeSplits)
{
    auto makeMask = [](auto a, auto b)
    {
        if constexpr (nwt == 2) // NOLINT
        {
            SplitType ret;
            ret[0] = a;
            ret[1] = b;
            return ret;
        }
        else { return SplitType{GpuConfig::ThreadMask(b << 32) + a}; } // NOLINT
    };

    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 64);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(1, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 1);
        EXPECT_EQ(splitLengths[1], 63);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0, 1u << 30);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 63);
        EXPECT_EQ(splitLengths[1], 1);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(2, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 2);
        EXPECT_EQ(splitLengths[1], 62);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(3, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 1);
        EXPECT_EQ(splitLengths[1], 1);
        EXPECT_EQ(splitLengths[2], 62);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(1u << 31, 1);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 32);
        EXPECT_EQ(splitLengths[1], 1);
        EXPECT_EQ(splitLengths[2], 31);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0, 8);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 36);
        EXPECT_EQ(splitLengths[1], 28);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0xFFFFFFFFu, 0x6FFFFFFFu);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        for (int i = 0; i < targetSize - 1; ++i)
        {
            if (i == 60) { EXPECT_EQ(splitLengths[i], 2); }
            else { EXPECT_EQ(splitLengths[i], 1); }
        }
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0xFFFFFFFF, 0x7FFFFFFF);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        for (int i = 0; i < targetSize - 1; ++i)
        {
            EXPECT_EQ(splitLengths[i], 1);
        }
    }
}

TEST(TargetGroups, groupVolumes)
{
    using T                        = double;
    using KeyType                  = uint64_t;
    constexpr LocalIndex groupSize = 64;

    LocalIndex first        = 4;
    LocalIndex last         = 128;
    LocalIndex numParticles = last - first;
    LocalIndex numGroups    = iceil(numParticles, groupSize);

    Box<T> box(0, last);
    double distCrit = std::cbrt(box.lx() * box.ly() * box.lz() / 64);

    auto leaves = OctreeMaker<KeyType>{}.divide().divide(2).makeTree();
    // nodeIdx                   0  1 |2  3  4  5  6   7  8  9 |10  11  12 13 14 15
    // fixed groups                |                 |                   |
    std::vector<unsigned> counts{4, 1, 8, 8, 8, 8, 31, 8, 8, 8, 16, 16, 16, 0, 0};
    std::vector<LocalIndex> layout(counts.size() + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    // these coordinates do not lie in the leaf cells specified by layout, but this is irrelevant for this test case
    thrust::device_vector<T> x(last), y(last), z(last), h(last);
    thrust::sequence(x.begin(), x.end(), 0);
    thrust::sequence(y.begin(), y.end(), 0);
    thrust::sequence(z.begin(), z.end(), 0);

    // introduce a split by increasing distance between particles 5 and 6
    x[5] -= 0.01;
    y[5] -= 0.01;
    z[5] -= 0.01;

    thrust::device_vector<LocalIndex> groupDiv(numGroups);
    thrust::device_vector<SplitType> splitMasks(numGroups);

    thrust::device_vector<KeyType> d_leaves    = leaves;
    thrust::device_vector<LocalIndex> d_layout = layout;

    unsigned numThreads = 256;
    unsigned gridSize   = numGroups * GpuConfig::warpSize;
    {
        float tolFactor = std::sqrt(3.0) / distCrit * 1.01;
        groupSplitsKernel<groupSize, T><<<iceil(gridSize, numThreads), numThreads>>>(
            first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves), nNodes(leaves), rawPtr(d_layout),
            box, tolFactor, rawPtr(splitMasks), rawPtr(groupDiv), numGroups);

        thrust::host_vector<LocalIndex> h_groupDiv = groupDiv;
        thrust::host_vector<LocalIndex> ref        = std::vector<LocalIndex>{2, 1};
        EXPECT_EQ(h_groupDiv, ref);
    }
    {
        float tolFactor = std::sqrt(3.0) / distCrit * 0.99;
        groupSplitsKernel<groupSize, T><<<iceil(gridSize, numThreads), numThreads>>>(
            first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves), nNodes(leaves), rawPtr(d_layout),
            box, tolFactor, rawPtr(splitMasks), rawPtr(groupDiv), numGroups);

        thrust::host_vector<LocalIndex> h_groupDiv = groupDiv;
        thrust::host_vector<LocalIndex> ref        = std::vector<LocalIndex>{64, 60};
        EXPECT_EQ(h_groupDiv, ref);
    }

    {
        thrust::device_vector<SplitType> S;
        thrust::device_vector<LocalIndex> temp, groups;

        float tolFactor = std::sqrt(3.0) / distCrit * 1.01;
        computeGroupSplits<groupSize>(first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves),
                                      nNodes(leaves), rawPtr(d_layout), box, tolFactor, S, temp, groups);

        EXPECT_EQ(groups.size(), 4);
        EXPECT_EQ(groups[0], 4);
        EXPECT_EQ(groups[1], 6);
        EXPECT_EQ(groups[2], 68);
        EXPECT_EQ(groups[3], 128);
    }
}

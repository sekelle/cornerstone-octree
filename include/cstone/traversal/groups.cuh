/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Particle target grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

//! @brief simple-fixed width group targets
template<class T>
__global__ void groupTargets(LocalIndex first,
                             LocalIndex last,
                             const T* x,
                             const T* y,
                             const T* z,
                             const T* h,
                             unsigned groupSize,
                             LocalIndex* groups,
                             unsigned numGroups)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    LocalIndex groupIdx = tid / groupSize;
    if (groupIdx >= numGroups) { return; }

    if (tid == groupIdx * groupSize)
    {
        LocalIndex scan  = first + groupIdx * groupSize;
        groups[groupIdx] = scan;
        if (groupIdx + 1 == numGroups) { groups[numGroups] = last; }
    }
}

/*! @brief Computes splitting patterns of target particle groups
 *
 * @tparam    T            float or double
 * @tparam    N            number of particles per thread
 * @param[in] pos          particle input
 * @param[in] distCritSq   maximum allowed distance^2 between two consecutive particles
 * @return                 a thread mask indicating the lanes between splits with 1-bits
 *                         all lanes return the same result
 */
template<class T, size_t N>
__device__ util::array<GpuConfig::ThreadMask, N> findSplits(util::array<Vec4<T>, N> pos, T distCritSq)
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    util::array<Vec3<T>, N> Xlane;
    for (int k = 0; k < N; ++k)
    {
        Xlane[k] = {pos[k][0], pos[k][1], pos[k][2]};
    }

    util::array<Vec3<T>, N> Xnext = Xlane;
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            Xnext[k][j] = shflDownSync(Xlane[k][j], 1);
        }
        if (k < N - 1)
        {
            Vec3<T> spill;
            for (int j = 0; j < 3; ++j)
            {
                spill[j] = shflSync(Xlane[k + 1][j], 0);
            }
            if (laneIdx == GpuConfig::warpSize - 1) { Xnext[k] = spill; }
        }
    }

    // The last difference in the warp (last lane of last segment N-1) will always be zero,
    // therefore the MSB of the return value will always be zero.
    util::array<GpuConfig::ThreadMask, N> splits;
    for (int k = 0; k < N; ++k)
    {
        T distSq   = norm2(Xnext[k] - Xlane[k]);
        bool split = distSq > distCritSq;
        splits[k]  = ballotSync(split);
    }

    return splits;
}

/*! @brief Count number of zero bits between 1-bits for the bitstream defined by @p splits
 *
 * @tparam N                  number of particles per thread
 * @param[in]  split          the split-bitmask
 * @param[out] splitLengths   output number of 0-bits between 1-bits, with space for popCount(split) + 1 elements
 *
 * Examples, assuming split is a 64-bit stream (either 2x32bit or 1x64bit):
 *    split = 0x0 --> splitLengths = {64}
 *    split = 0x1 --> splitLengths = {1, 63}
 *    split = 0x2 --> splitLengths = {2, 62}
 *    split = 0x3 --> splitLengths = {1, 1, 61}
 */
template<size_t N>
__device__ void makeSplits(util::array<GpuConfig::ThreadMask, N> split, LocalIndex* splitLengths)
{
    auto readSegment = [&splitLengths](GpuConfig::ThreadMask mask, int carry)
    {
        int bitsRemaining = CHAR_BIT * sizeof(GpuConfig::ThreadMask);
        while (popCount(mask))
        {
            int length = countTrailingZeros(mask) + 1;
            bitsRemaining -= length;
            *splitLengths++ = length + carry;
            carry = 0;
            mask >>= length;
        }
        return carry + bitsRemaining;
    };

    int carry = 0;
    for (int k = 0; k < N; ++k)
    {
        carry = readSegment(split[k], carry);
    }
    *splitLengths = carry;
}

template<size_t N>
__global__ void makeSplitsKernel(const util::array<GpuConfig::ThreadMask, N>* splitMasks,
                                 const LocalIndex* groupOffsets,
                                 LocalIndex numFixedGroups,
                                 LocalIndex* splitLengths)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numFixedGroups) { return; }

    makeSplits(splitMasks[tid], splitLengths + groupOffsets[tid]);
}

/*! @brief compute split pattern and split counts for each initial fixed-size target particle group
 *
 * @tparam groupSize           a multiple of GpuConfig::warpSize
 * @tparam T                   float or double
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  first           first particle from x,y,z,h to consider
 * @param[in]  last            last particle to consider
 * @param[in]  x               local x particle coordinates
 * @param[in]  y               local y particle coordinates
 * @param[in]  z               local z particle coordinates
 * @param[in]  h               local smoothing lengths
 * @param[in]  leaves          cornerstone leaf-cell array
 * @param[in]  numLeaves       number of leaves in @p leaves
 * @param[in]  layout          layout[i] is the x,y,z,h-array particle index of the first particle in leaf i,
 * @param[in]  box             global coordinate bounding box
 * @param[in]  tolFactor       max distance between consecutive particles is
 *                             tolFactor * cbrt(smallest leaf node size in group)
 * @param[out] splitMasks      split mask for each of the ceil((last-first)/groupSize) fixed-size groups
 * @param[out] numSplitsPerGroup 1 + number-of-1-bits in @p splitMasks for each fixed-size group
 */
template<unsigned groupSize, class T, class KeyType>
__global__ void groupSplitsKernel(LocalIndex first,
                                  LocalIndex last,
                                  const T* x,
                                  const T* y,
                                  const T* z,
                                  const T* h,
                                  const KeyType* leaves,
                                  TreeNodeIndex numLeaves,
                                  const LocalIndex* layout,
                                  const Box<T> box,
                                  float tolFactor,
                                  util::array<GpuConfig::ThreadMask, groupSize / GpuConfig::warpSize>* splitMasks,
                                  LocalIndex* numSplitsPerGroup,
                                  unsigned /*numGroups*/)
{
    constexpr LocalIndex nwt = groupSize / GpuConfig::warpSize;

    LocalIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    LocalIndex warpIdx = tid / GpuConfig::warpSize;
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    if (warpIdx * groupSize >= last) { return; }

    LocalIndex bodyIdx[nwt];
    for (int k = 0; k < nwt; ++k)
    {
        bodyIdx[k] = imin(first + warpIdx * groupSize + k * GpuConfig::warpSize + laneIdx, last - 1);
    }

    TreeNodeIndex leafIdx[nwt];
    for (int k = 0; k < nwt; ++k)
    {
        leafIdx[k] = stl::upper_bound(layout, layout + numLeaves, bodyIdx[k]) - layout - 1;
    }

    Box<T> unitBox(0, 1);
    T nodeVolume = 1;
    for (int k = 0; k < nwt; ++k)
    {
        auto [nodeCenter, nodeSize] =
            centerAndSize<KeyType>(sfcIBox(sfcKey(leaves[leafIdx[0]]), sfcKey(leaves[leafIdx[0] + 1])), unitBox);
        T vol      = 8 * nodeSize[0] * nodeSize[1] * nodeSize[2];
        nodeVolume = min(vol, nodeVolume);
    }
    nodeVolume = warpMin(nodeVolume);
    T distCrit = std::cbrt(nodeVolume) * tolFactor;

    // load target coordinates
    util::array<Vec4<T>, nwt> pos_i;
    for (int k = 0; k < nwt; k++)
    {
        pos_i[k] = {x[bodyIdx[k]] * box.ilx(), y[bodyIdx[k]] * box.ily(), z[bodyIdx[k]] * box.ilz(),
                    h ? T(2) * h[bodyIdx[k]] : T(0)};
    }

    auto splitMask = findSplits(pos_i, distCrit * distCrit);

    if (laneIdx == 0)
    {
        splitMasks[warpIdx] = splitMask;
        int numSubGroups    = 1;
        for (int k = 0; k < nwt; ++k)
        {
            numSubGroups += popCount(splitMask[k]);
        }
        numSplitsPerGroup[warpIdx] = numSubGroups;
    }
}

//! @brief convenience wrapper for groupSplitsKernel
template<unsigned groupSize, class T, class KeyType>
void computeGroupSplits(
    LocalIndex first,
    LocalIndex last,
    const T* x,
    const T* y,
    const T* z,
    const T* h,
    const KeyType* leaves,
    TreeNodeIndex numLeaves,
    const LocalIndex* layout,
    const Box<T> box,
    float tolFactor,
    thrust::device_vector<util::array<GpuConfig::ThreadMask, groupSize / GpuConfig::warpSize>>& splitMasks,
    thrust::device_vector<LocalIndex>& numSplitsPerGroup,
    thrust::device_vector<LocalIndex>& groups)
{
    LocalIndex numParticles   = last - first;
    LocalIndex numFixedGroups = iceil(numParticles, groupSize);
    unsigned numThreads       = 256;
    unsigned gridSize         = numFixedGroups * GpuConfig::warpSize;

    splitMasks.resize(numFixedGroups);

    numSplitsPerGroup.reserve(numFixedGroups * 1.1);
    numSplitsPerGroup.resize(numFixedGroups);

    groupSplitsKernel<groupSize, T><<<iceil(gridSize, numThreads), numThreads>>>(
        first, last, x, y, z, h, leaves, numLeaves, layout, box, tolFactor, rawPtr(splitMasks),
        rawPtr(numSplitsPerGroup), numFixedGroups);

    groups.reserve(numFixedGroups * 1.1);
    groups.resize(numFixedGroups + 1);
    exclusiveScanGpu(rawPtr(numSplitsPerGroup), rawPtr(numSplitsPerGroup) + numFixedGroups + 1, rawPtr(groups));
    LocalIndex newNumGroups = groups.back();

    auto& newGroupSizes = numSplitsPerGroup;
    newGroupSizes.resize(newNumGroups);

    makeSplitsKernel<<<numFixedGroups, numThreads>>>(rawPtr(splitMasks), rawPtr(groups), numFixedGroups,
                                                     rawPtr(newGroupSizes));

    groups.resize(newNumGroups + 1);
    exclusiveScanGpu(rawPtr(newGroupSizes), rawPtr(newGroupSizes) + newNumGroups + 1, rawPtr(groups), first);
    groups.back() = last;
}

} // namespace cstone
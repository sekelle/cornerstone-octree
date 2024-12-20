/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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

#include <stdexcept>

#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/groups_gpu.cuh"

namespace cstone
{

//! @brief simple-fixed width group targets
__global__ static void
fixedGroupsKernel(LocalIndex first, LocalIndex last, unsigned groupSize, LocalIndex* groups, unsigned numGroups)
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

void computeFixedGroups(LocalIndex first, LocalIndex last, unsigned groupSize, GroupData<GpuTag>& groups)
{
    LocalIndex numBodies = last - first;
    LocalIndex numGroups = iceil(numBodies, groupSize);
    groups.data.resize(numGroups + 1);
    fixedGroupsKernel<<<iceil(numBodies, 256), 256>>>(first, last, groupSize, rawPtr(groups.data), numGroups);

    groups.firstBody  = first;
    groups.lastBody   = last;
    groups.numGroups  = numGroups;
    groups.groupStart = rawPtr(groups.data);
    groups.groupEnd   = rawPtr(groups.data) + 1;
}

//! @brief convenience wrapper for groupSplitsKernel
template<unsigned groupSize, class Tc, class T, class KeyType>
void computeGroupSplitsImpl(
    LocalIndex first,
    LocalIndex last,
    const Tc* x,
    const Tc* y,
    const Tc* z,
    const T* h,
    const KeyType* leaves,
    TreeNodeIndex numLeaves,
    const LocalIndex* layout,
    const Box<Tc> box,
    float tolFactor,
    DeviceVector<util::array<GpuConfig::ThreadMask, groupSize / GpuConfig::warpSize>>& splitMasks,
    DeviceVector<LocalIndex>& numSplitsPerGroup,
    DeviceVector<LocalIndex>& groups)
{
    LocalIndex numParticles   = last - first;
    LocalIndex numFixedGroups = iceil(numParticles, groupSize);
    unsigned numThreads       = 256;
    unsigned gridSize         = numFixedGroups * GpuConfig::warpSize;

    splitMasks.resize(numFixedGroups);

    numSplitsPerGroup.reserve(numFixedGroups * 1.1);
    numSplitsPerGroup.resize(numFixedGroups);

    groupSplitsKernel<groupSize><<<iceil(gridSize, numThreads), numThreads>>>(
        first, last, x, y, z, h, leaves, numLeaves, layout, box, tolFactor, rawPtr(splitMasks),
        rawPtr(numSplitsPerGroup), numFixedGroups);

    groups.reserve(numFixedGroups * 1.1);
    groups.resize(numFixedGroups + 1);
    exclusiveScanGpu(rawPtr(numSplitsPerGroup), rawPtr(numSplitsPerGroup) + numFixedGroups + 1, rawPtr(groups));
    LocalIndex newNumGroups;
    memcpyD2H(rawPtr(groups) + groups.size() - 1, 1, &newNumGroups);

    auto& newGroupSizes = numSplitsPerGroup;
    newGroupSizes.resize(newNumGroups);

    makeSplitsKernel<<<numFixedGroups, numThreads>>>(rawPtr(splitMasks), rawPtr(groups), numFixedGroups,
                                                     rawPtr(newGroupSizes));

    groups.resize(newNumGroups + 1);
    exclusiveScanGpu(rawPtr(newGroupSizes), rawPtr(newGroupSizes) + newNumGroups + 1, rawPtr(groups), first);
    memcpyH2D(&last, 1, rawPtr(groups) + groups.size() - 1);
}

template<class Tc, class T, class KeyType>
void computeGroupSplits(LocalIndex first,
                        LocalIndex last,
                        const Tc* x,
                        const Tc* y,
                        const Tc* z,
                        const T* h,
                        const KeyType* leaves,
                        TreeNodeIndex numLeaves,
                        const LocalIndex* layout,
                        const Box<Tc> box,
                        unsigned groupSize,
                        float tolFactor,
                        DeviceVector<LocalIndex>& numSplitsPerGroup,
                        DeviceVector<LocalIndex>& groups)
{
    if (groupSize == GpuConfig::warpSize)
    {
        DeviceVector<util::array<GpuConfig::ThreadMask, 1>> splitMasks;
        computeGroupSplitsImpl<GpuConfig::warpSize>(first, last, x, y, z, h, leaves, numLeaves, layout, box, tolFactor,
                                                    splitMasks, numSplitsPerGroup, groups);
    }
    else if (groupSize == 2 * GpuConfig::warpSize)
    {
        DeviceVector<util::array<GpuConfig::ThreadMask, 2>> splitMasks;
        computeGroupSplitsImpl<2 * GpuConfig::warpSize>(first, last, x, y, z, h, leaves, numLeaves, layout, box,
                                                        tolFactor, splitMasks, numSplitsPerGroup, groups);
    }
    else { throw std::runtime_error("Unsupported spatial group size\n"); }
}

#define COMPUTE_GROUP_SPLITS(Tc, T, KeyType)                                                                           \
    template void computeGroupSplits(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,         \
                                     const T* h, const KeyType* leaves, TreeNodeIndex numLeaves,                       \
                                     const LocalIndex* layout, const Box<Tc> box, unsigned groupSize, float tolFactor, \
                                     DeviceVector<LocalIndex>& numSplitsPerGroup, DeviceVector<LocalIndex>& groups);

COMPUTE_GROUP_SPLITS(double, double, uint64_t);
COMPUTE_GROUP_SPLITS(double, float, uint64_t);
COMPUTE_GROUP_SPLITS(float, float, uint64_t);

} // namespace cstone

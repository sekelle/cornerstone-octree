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
 * @brief Utility functions for determining the layout of particle buffers on a given rank
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank will be assigned a part of the SFC, equating to one or multiple ranges of
 * node indices of the global cornerstone octree. In a addition to the assigned nodes,
 * each rank must also store particle data for those nodes in the global octree which are
 * halos of the assigned nodes. Both types of nodes present on the rank are stored in the same
 * particle array (x,y,z,h,...) according to increasing node index, which is the same
 * as increasing Morton code.
 *
 * Given
 *  - the global cornerstone tree
 *  - its assignment to ranks
 *  - lists of in/outgoing halo nodes (global indices) per rank,
 * the utility functions in this file determine the position and size of each node (halo or assigned node)
 * in the particle buffers. The resulting layout is valid for all particle buffers, such as x,y,z,h,d,p,...
 *
 * Note:
 * If a node of the global cornerstone octree has index i, this means its Morton code range is tree[i] - tree[i+1]
 */

#pragma once

#include <vector>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/halos/discovery.hpp"

namespace cstone
{

/*! @brief calculate the location (offset) of each focus tree leaf node in the particle arrays
 *
 * @param focusLeafCounts   node counts of the focus leaves, size N
 * @param haloFlags         flag for each node, with a non-zero value if present as halo node, size N
 * @param firstAssignedIdx  first focus leaf idx to treat as part of the assigned nodes on the executing rank
 * @param lastAssignedIdx   last focus leaf idx to treat as part of the assigned nodes on the executing rank
 * @return                  array with offsets, size N+1. The first element is zero, the last element is
 *                          equal to the sum of all all present (assigned+halo) node counts.
 */
inline
std::vector<LocalParticleIndex> computeNodeLayout(gsl::span<const unsigned> focusLeafCounts,
                                                  gsl::span<const int> haloFlags,
                                                  TreeNodeIndex firstAssignedIdx,
                                                  TreeNodeIndex lastAssignedIdx)
{
    std::vector<LocalParticleIndex> layout(focusLeafCounts.size() + 1, 0);

    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < focusLeafCounts.size(); ++i)
    {
        bool onRank = firstAssignedIdx <= i && i < lastAssignedIdx;
        if (onRank || haloFlags[i]) { layout[i] = focusLeafCounts[i]; }
    }

    exclusiveScan(layout.data(), layout.size());

    return layout;
}

/*! @brief computes a list which local array ranges are going to be filled with halo particles
 *
 * @param layout       prefix sum of leaf counts of locally present nodes (see computeNodeLayout)
 *                     length N+1
 * @param haloFlags    0 or 1 for each leaf, length N
 * @param assignment   assignment of leaf nodes to peer ranks
 * @param peerRanks    list of peer ranks
 * @return             list of array index ranges for the receiving part in exchangeHalos
 */
inline
SendList computeHaloReceiveList(gsl::span<const LocalParticleIndex> layout,
                                gsl::span<const int> haloFlags,
                                gsl::span<const TreeIndexPair> assignment,
                                gsl::span<const int> peerRanks)
{
    SendList ret(assignment.size());

    for (int peer : peerRanks)
    {
        TreeNodeIndex peerStartIdx = assignment[peer].start();
        TreeNodeIndex peerEndIdx   = assignment[peer].end();

        std::vector<LocalParticleIndex> receiveRanges =
            extractMarkedElements<LocalParticleIndex>(layout, haloFlags, peerStartIdx, peerEndIdx);

        for (int i = 0; i < receiveRanges.size(); i +=2 )
        {
            ret[peer].addRange(receiveRanges[i], receiveRanges[i+1]);
        }
    }

    return ret;
}

template<class... Arrays>
void relocate(LocalParticleIndex newSize, LocalParticleIndex offset, Arrays&... arrays)
{
    std::array data{(&arrays)...};

    using ArrayType = std::decay_t<decltype(*data[0])>;

    for (auto arrayPtr : data)
    {
        ArrayType newArray(newSize);
        std::copy(arrayPtr->begin(), arrayPtr->end(), newArray.begin() + offset);
        swap(newArray, *arrayPtr);
    }
}


//! @brief reallocate arrays to the specified size
template<class... Arrays>
void reallocate(std::size_t size, Arrays&... arrays)
{
    std::array data{ (&arrays)... };

    size_t current_capacity = data[0]->capacity();
    if (size > current_capacity)
    {
        // limit reallocation growth to 5% instead of 200%
        auto reserve_size = static_cast<size_t>(double(size) * 1.05);
        for (auto array : data)
        {
            array->reserve(reserve_size);
        }
    }

    for (auto array : data)
    {
        array->resize(size);
    }
}

} // namespace cstone

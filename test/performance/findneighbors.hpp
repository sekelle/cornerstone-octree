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
 * @brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cmath>

#include "box.hpp"
#include "definitions.h"

namespace cstone
{

//! @brief generic depth-first traversal of an octree that works on CPU and GPU with customizable descent criteria
template<class C, class A>
HOST_DEVICE_FUN void
depthFirstTraversal(const TreeNodeIndex* childOffsets, C&& continuationCriterion, A&& endpointAction)
{
    bool descend = continuationCriterion(0);
    if (!descend) return;

    if (childOffsets[0] == 0)
    {
        // root node is already the endpoint
        endpointAction(0);
        return;
    }

    constexpr int maxStackDepth = 64;
    TreeNodeIndex stack[maxStackDepth];
    stack[0] = 0;

    TreeNodeIndex stackPos = 1;
    TreeNodeIndex node     = 0; // start at the root

    do
    {
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = childOffsets[node] + octant;
            bool descend        = continuationCriterion(child);
            if (descend)
            {
                if (childOffsets[child] == 0)
                {
                    // endpoint reached with child is a leaf node
                    endpointAction(child);
                }
                else
                {
                    assert(stackPos < maxStackDepth);
                    stack[stackPos++] = child; // push
                }
            }
        }
        node = stack[--stackPos];

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

/*! @brief findNeighbors of particle number @p id within a radius. Works on CPU and GPU.
 *
 * @tparam     T               coordinate type, float or double
 * @tparam     KeyType         32- or 64-bit Morton or Hilbert key type
 * @param[in]  i               the index of the particle for which to look for neighbors
 * @param[in]  x               particle x-coordinates in SFC order (as indexed by @p tree.layout)
 * @param[in]  y               particle y-coordinates in SFC order
 * @param[in]  z               particle z-coordinates in SFC order
 * @param[in]  h               smoothing lengths (1/2 the search radius) in SFC order
 * @param[in]  tree            octree connectivity and particle indexing
 * @param[in]  box             coordinate bounding box that was used to calculate the Morton codes
 * @param[in]  ngmax           maximum number of neighbors per particle
 * @param[out] neighbors       output to store the neighbors
 * @return                     neighbor count of particle @p i
 */
template<class T, class KeyType>
HOST_DEVICE_FUN unsigned findNeighbors(LocalIndex i,
                                       const T* x,
                                       const T* y,
                                       const T* z,
                                       const T* h,
                                       const OctreeNsView<T, KeyType>& tree,
                                       const Box<T>& box,
                                       unsigned ngmax,
                                       LocalIndex* neighbors)
{
    Vec3<T> target{x[i], y[i], z[i]};
    T hi                  = h[i];
    T radiusSq            = 4.0 * hi * hi;
    unsigned numNeighbors = 0;

    auto overlaps = [target, radiusSq, centers = tree.centers, sizes = tree.sizes, &box](TreeNodeIndex idx)
    {
        auto nodeCenter = centers[idx];
        auto nodeSize   = sizes[idx];
        return norm2(minDistance(target, nodeCenter, nodeSize, box)) < radiusSq;
    };

    auto searchBox = [i, target, radiusSq, &tree, x, y, z, ngmax, neighbors, &numNeighbors](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        LocalIndex firstParticle = tree.layout[leafIdx];
        LocalIndex lastParticle  = tree.layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (norm2(Vec3<T>{x[j], y[j], z[j]} - target) < radiusSq)
            {
                if (numNeighbors < ngmax) { neighbors[numNeighbors] = j; }
                numNeighbors++;
            }
        }
    };

    depthFirstTraversal(tree.childOffsets, overlaps, searchBox);

    return numNeighbors;
}

//! @brief OpenMP parallel CPU version of neighbor search
template<class T, class KeyType>
void findNeighbors(const T* x,
                   const T* y,
                   const T* z,
                   const T* h,
                   LocalIndex firstId,
                   LocalIndex lastId,
                   const Box<T>& box,
                   const OctreeNsView<T, KeyType>& treeView,
                   unsigned ngmax,
                   LocalIndex* neighbors,
                   unsigned* neighborsCount)
{
    LocalIndex numWork = lastId - firstId;

#pragma omp parallel for
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id     = i + firstId;
        neighborsCount[i] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
    }
}

} // namespace cstone

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
#include "traversal.hpp"

namespace cstone
{

//! @brief compute geometric node centers based on node SFC keys and the global bounding box
template<class KeyType, class T>
void nodeFpCenters(const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i)
    {
        KeyType prefix                  = prefixes[i];
        KeyType startKey                = decodePlaceholderBit(prefix);
        unsigned level                  = decodePrefixLength(prefix) / 3;
        auto nodeBox                    = sfcIBox(sfcKey(startKey), level);
        util::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
    }
}

//! @brief compute squared distance between to points in 3D
template<class T>
HOST_DEVICE_FUN constexpr T distanceSq(T x1, T y1, T z1, T x2, T y2, T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    return xx * xx + yy * yy + zz * zz;
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
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];

    T radiusSq = 4.0 * hi * hi;
    Vec3<T> particle{xi, yi, zi};
    unsigned numNeighbors = 0;

    auto overlaps = [particle, radiusSq, centers = tree.centers, sizes = tree.sizes, &box](TreeNodeIndex idx)
    {
        auto nodeCenter = centers[idx];
        auto nodeSize   = sizes[idx];
        return norm2(minDistance(particle, nodeCenter, nodeSize, box)) < radiusSq;
    };

    auto searchBox = [i, particle, radiusSq, &tree, x, y, z, ngmax, neighbors, &numNeighbors](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        LocalIndex firstParticle = tree.layout[leafIdx];
        LocalIndex lastParticle  = tree.layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (distanceSq(x[j], y[j], z[j], particle[0], particle[1], particle[2]) < radiusSq)
            {
                if (numNeighbors < ngmax) { neighbors[numNeighbors] = j; }
                numNeighbors++;
            }
        }
    };

    singleTraversal(tree.childOffsets, overlaps, searchBox);

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

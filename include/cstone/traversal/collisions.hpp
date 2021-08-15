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
 * @brief  Collision detection for halo discovery using octree traversal
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/traversal.hpp"

namespace cstone
{

template<class KeyType, class F>
void findCollisions(const Octree<KeyType>& octree,
                    F&& endpointAction,
                    const IBox& target,
                    KeyType excludeStart,
                    KeyType excludeEnd)
{
    auto overlaps = [excludeStart, excludeEnd, &octree, &target](TreeNodeIndex idx)
    {
      KeyType nodeKey = octree.codeStart(idx);
      int level = octree.level(idx);
      IBox sourceBox = hilbertIBox(nodeKey, level);
      return !containedIn(nodeKey, nodeKey + nodeRange<KeyType>(level), excludeStart, excludeEnd)
             && overlap<KeyType>(sourceBox, target);
    };

    singleTraversal(octree, overlaps, endpointAction);
}

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType        float or double
 * @param[in]  octree            fully linked octree
 * @param[in]  interactionRadii  effective halo search radii per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first leaf node index of @p leaves to consider as local
 * @param[in]  lastNode          last leaf node index of @p leaves to consider as local
 * @param[out] collisionFlags    array of length nNodes(leaves), each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 */
template<class KeyType, class RadiusType, class CoordinateType>
void findHalos(const Octree<KeyType>& octree,
               const RadiusType* interactionRadii,
               const Box<CoordinateType>& box,
               TreeNodeIndex firstNode,
               TreeNodeIndex lastNode,
               int* collisionFlags)
{
    auto leaves = octree.treeLeaves();
    KeyType lowestCode  = leaves[firstNode];
    KeyType highestCode = leaves[lastNode];

    auto markCollisions = [flags = collisionFlags](TreeNodeIndex i) { flags[i] = 1; };

    // loop over all the nodes in range
    #pragma omp parallel for
    for (TreeNodeIndex nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
    {
        RadiusType radius = interactionRadii[nodeIdx];
        IBox haloBox      = makeHaloBox(leaves[nodeIdx], leaves[nodeIdx + 1], radius, box);

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestCode, highestCode, haloBox)) { continue; }

        findCollisions(octree, markCollisions, haloBox, lowestCode, highestCode);
    }
}

} // namespace cstone
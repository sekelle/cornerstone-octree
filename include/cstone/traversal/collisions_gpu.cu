/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  GPU driver for halo discovery using traversal of an octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/math.hpp"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/traversal/macs.hpp"

namespace cstone
{

template<class KeyType, class RadiusType, class T>
__global__ void findHalosKernel(const KeyType* nodePrefixes,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                const TreeNodeIndex* internalToLeaf,
                                const KeyType* leaves,
                                const RadiusType* interactionRadii,
                                const Box<T> box,
                                TreeNodeIndex firstNode,
                                TreeNodeIndex lastNode,
                                uint8_t* collisionFlags)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x + firstNode;

    auto markCollisions = [collisionFlags, internalToLeaf](TreeNodeIndex i) { collisionFlags[internalToLeaf[i]] = 1; };

    if (leafIdx < lastNode)
    {
        RadiusType radius  = interactionRadii[leafIdx];
        IBox haloBox       = makeHaloBox(leaves[leafIdx], leaves[leafIdx + 1], radius, box);
        KeyType lowestKey  = leaves[firstNode];
        KeyType highestKey = leaves[lastNode];

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestKey, highestKey, haloBox)) { return; }

        // mark all colliding node indices outside [lowestKey:highestKey]
        findCollisions(nodePrefixes, childOffsets, parents, markCollisions, haloBox, lowestKey, highestKey);
    }
}

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam T                     float or double
 * @param[in]  prefixes          Warren-Salmon node keys of the octree, length = numTreeNodes
 * @param[in]  childOffsets      child offsets array, length = numTreeNodes
 * @param[in]  internalToLeaf    map leaf node indices of fully linked format to cornerstone order
 * @param[in]  leaves            cstone array of leaf node keys
 * @param[in]  interactionRadii  effective halo search radii per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first cstone leaf node index to consider as local
 * @param[in]  lastNode          last cstone leaf node index to consider as local
 * @param[out] collisionFlags    array of length numLeafNodes, each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 */
template<class KeyType, class RadiusType, class T>
void findHalosGpu(const KeyType* prefixes,
                  const TreeNodeIndex* childOffsets,
                  const TreeNodeIndex* parents,
                  const TreeNodeIndex* internalToLeaf,
                  const KeyType* leaves,
                  const RadiusType* interactionRadii,
                  const Box<T>& box,
                  TreeNodeIndex firstNode,
                  TreeNodeIndex lastNode,
                  uint8_t* collisionFlags)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(lastNode - firstNode, numThreads);

    if (numBlocks == 0) { return; }
    findHalosKernel<<<numBlocks, numThreads>>>(prefixes, childOffsets, parents, internalToLeaf, leaves,
                                               interactionRadii, box, firstNode, lastNode, collisionFlags);
}

#define FIND_HALOS_GPU(KeyType, RadiusType, T)                                                                         \
    template void findHalosGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,                             \
                               const TreeNodeIndex* parents, const TreeNodeIndex* internalToLeaf,                      \
                               const KeyType* leaves, const RadiusType* interactionRadii, const Box<T>& box,           \
                               TreeNodeIndex firstNode, TreeNodeIndex lastNode, uint8_t* collisionFlags)

FIND_HALOS_GPU(uint32_t, float, float);
FIND_HALOS_GPU(uint32_t, float, double);
FIND_HALOS_GPU(uint64_t, float, float);
FIND_HALOS_GPU(uint64_t, float, double);

template<class T, class KeyType>
__global__ void markMacsGpuKernel(const KeyType* prefixes,
                                  const TreeNodeIndex* childOffsets,
                                  const TreeNodeIndex* parents,
                                  const Vec4<T>* centers,
                                  const Box<T> box,
                                  const KeyType* focusNodes,
                                  TreeNodeIndex numFocusNodes,
                                  bool limitSource,
                                  uint8_t* markings)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numFocusNodes) { return; }

    KeyType focusStart = focusNodes[0];
    KeyType focusEnd   = focusNodes[numFocusNodes];

    IBox target    = sfcIBox(sfcKey(focusNodes[tid]), sfcKey(focusNodes[tid + 1]));
    IBox targetExt = IBox(target.xmin() - 1, target.xmax() + 1, target.ymin() - 1, target.ymax() + 1, target.zmin() - 1,
                          target.zmax() + 1);
    if (containedIn(focusStart, focusEnd, targetExt)) { return; }

    auto [targetCenter, targetSize] = centerAndSize<KeyType>(target, box);
    unsigned maxLevel               = maxTreeLevel<KeyType>{};
    if (limitSource) { maxLevel = stl::max(int(treeLevel(focusNodes[tid + 1] - focusNodes[tid])) - 1, 0); }
    markMacPerBox(targetCenter, targetSize, maxLevel, prefixes, childOffsets, parents, centers, box, focusStart,
                  focusEnd, markings);
}

template<class T, class KeyType>
void markMacsGpu(const KeyType* prefixes,
                 const TreeNodeIndex* childOffsets,
                 const TreeNodeIndex* parents,
                 const Vec4<T>* centers,
                 const Box<T>& box,
                 const KeyType* focusNodes,
                 TreeNodeIndex numFocusNodes,
                 bool limitSource,
                 uint8_t* markings)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(numFocusNodes, numThreads);

    if (numFocusNodes)
    {
        markMacsGpuKernel<<<numBlocks, numThreads>>>(prefixes, childOffsets, parents, centers, box, focusNodes,
                                                     numFocusNodes, limitSource, markings);
    }
}

#define MARK_MACS_GPU(KeyType, T)                                                                                      \
    template void markMacsGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,                              \
                              const TreeNodeIndex* parents, const Vec4<T>* centers, const Box<T>& box,                 \
                              const KeyType* focusNodes, TreeNodeIndex numFocusNodes, bool limitSource,                \
                              uint8_t* markings)

MARK_MACS_GPU(uint64_t, double);
MARK_MACS_GPU(uint64_t, float);
MARK_MACS_GPU(unsigned, double);
MARK_MACS_GPU(unsigned, float);

} // namespace cstone

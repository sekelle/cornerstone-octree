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

#pragma once

#include "cstone/execution.hpp"
#include "cstone/traversal/collisions.hpp"

namespace cstone
{

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam T                     float or double
 * @param[in]  prefixes          Warren-Salmon node keys of the octree, length = numTreeNodes
 * @param[in]  childOffsets      child offsets array, length = numTreeNodes
 * @param[in]  parents           parent of each node i, stored at index (i-1)/8
 * @param[in]  nodeCenters       geometric center of each octree node
 * @param[in]  nodeSizes         geometric size of each octree node
 * @param[in]  leaves            cstone array of leaf node keys
 * @param[in]  searchCenters     effective halo search box center per octree (leaf) node, accessed [firstNode:lastNode]
 * @param[in]  searchSizes       effective halo search box size per octree (leaf) node, accessed [firstNode:lastNode]
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first cstone leaf node index to consider as local
 * @param[in]  lastNode          last cstone leaf node index to consider as local
 * @param[out] collisionFlags    array of length numLeafNodes, each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 * @param[in]  exec              execution policy
 */
template<class KeyType, class T>
extern void findHalosGpu(execution::Gpu exec,
                         const KeyType* prefixes,
                         const TreeNodeIndex* childOffsets,
                         const TreeNodeIndex* parents,
                         const Vec3<T>* nodeCenters,
                         const Vec3<T>* nodeSizes,
                         const KeyType* leaves,
                         const Vec3<T>* searchCenters,
                         const Vec3<T>* searchSizes,
                         const Box<T>& box,
                         TreeNodeIndex firstNode,
                         TreeNodeIndex lastNode,
                         uint8_t* collisionFlags);

/*! @brief mark the leaves that contain the given SFC keys as halos
 *
 * @param[in]  leaves          cstone array of leaf node keys, length numLeaves + 1
 * @param[in]  numLeaves       number of leaf nodes
 * @param[in]  leafToInternal  octree node index of each leaf, length numLeaves
 * @param[in]  parents         parent node index of each group of 8 siblings
 * @param[in]  firstNode       first leaf index assigned to the executing rank
 * @param[in]  lastNode        last leaf index assigned to the executing rank
 * @param[in]  keys            SFC keys, length numKeys
 * @param[in]  numKeys         number of keys
 * @param[out] collisionFlags  octree node flags, leaves outside [firstNode:lastNode] that contain a key and their
 *                             ancestors are set to 1, other flags are not touched
 */
template<class KeyType>
extern void markHaloKeysGpu(execution::Gpu exec,
                            const KeyType* leaves,
                            TreeNodeIndex numLeaves,
                            const TreeNodeIndex* leafToInternal,
                            const TreeNodeIndex* parents,
                            TreeNodeIndex firstNode,
                            TreeNodeIndex lastNode,
                            const KeyType* keys,
                            size_t numKeys,
                            uint8_t* collisionFlags);

template<class T, class KeyType>
extern void markMacsGpu(execution::Gpu exec,
                        const KeyType* prefixes,
                        const TreeNodeIndex* childOffsets,
                        const TreeNodeIndex* parents,
                        const Vec4<T>* centers,
                        const Box<T>& box,
                        const KeyType* focusNodes,
                        TreeNodeIndex numFocusNodes,
                        bool limitSource,
                        uint8_t* markings);

} // namespace cstone

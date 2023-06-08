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
 * @brief  Compute the internal part of a cornerstone octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * General algorithm:
 *      cornerstone octree (leaves) -> internal binary radix tree -> internal octree
 *
 * Like the cornerstone octree, the internal octree is stored in a linear memory layout
 * with tree nodes placed next to each other in a single buffer. Construction
 * is fully parallel and non-recursive and non-iterative. Traversal is possible non-recursively
 * in an iterative fashion with a local stack.
 */

#pragma once

#include <iterator>
#include <vector>

#include "annotation.hpp"
#include "cuda_utils.hpp"
#include "gather.hpp"
#include "sfc.hpp"
#include "stl.hpp"
#include "accel_switch.hpp"
#include "csarray.hpp"
#include "gsl-lite.hpp"

namespace cstone
{

/*! @brief map a binary node index to an octree node index
 *
 * @tparam KeyType    32- or 64-bit unsigned integer
 * @param  key        a cornerstone leaf cell key
 * @param  level      the subdivision level of @p key
 * @return            the index offset
 *
 * if
 *      - cstree is a cornerstone leaf array
 *      - l = commonPrefix(cstree[j], cstree[j+1]), l % 3 == 0
 *      - k = cstree[j]
 *
 * then i = (j + binaryKeyWeight(k, l) / 7 equals the index of the internal octree node with key k,
 * see unit test of this function for an illustration
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr TreeNodeIndex binaryKeyWeight(KeyType key, unsigned level)
{
    TreeNodeIndex ret = 0;
    for (unsigned l = 1; l <= level + 1; ++l)
    {
        unsigned digit = octalDigit(key, l);
        ret += digitWeight(digit);
    }
    return ret;
}

/*! @brief combine internal and leaf tree parts into a single array with the nodeKey prefixes
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  leaves            cornerstone SFC keys, length numLeafNodes + 1
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  numLeafNodes      total number of nodes
 * @param[in]  binaryToOct       translation map from binary to octree nodes
 * @param[out] prefixes          output octree SFC keys, length @p numInternalNodes + numLeafNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] internalToLeaf    iota 0,1,2,3,... sequence for later use, length same as @p prefixes
 */
template<class KeyType>
void createUnsortedLayoutCpu(const KeyType* leaves,
                             TreeNodeIndex numInternalNodes,
                             TreeNodeIndex numLeafNodes,
                             KeyType* prefixes,
                             TreeNodeIndex* internalToLeaf)
{
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex tid = 0; tid < numLeafNodes; ++tid)
    {
        KeyType key                            = leaves[tid];
        unsigned level                         = treeLevel(leaves[tid + 1] - key);
        prefixes[tid + numInternalNodes]       = encodePlaceholderBit(key, 3 * level);
        internalToLeaf[tid + numInternalNodes] = tid + numInternalNodes;

        unsigned prefixLength = commonPrefix(key, leaves[tid + 1]);
        if (prefixLength % 3 == 0 && tid < numLeafNodes - 1)
        {
            TreeNodeIndex octIndex   = (tid + binaryKeyWeight(key, prefixLength / 3)) / 7;
            prefixes[octIndex]       = encodePlaceholderBit(key, prefixLength);
            internalToLeaf[octIndex] = octIndex;
        }
    }
}

/*! @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  prefixes          octree node prefixes in Warren-Salmon format
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  leafToInternal    translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[in]  levelRange        indices of the first node at each level
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
template<class KeyType>
void linkTreeCpu(const KeyType* prefixes,
                 TreeNodeIndex numInternalNodes,
                 const TreeNodeIndex* leafToInternal,
                 const TreeNodeIndex* levelRange,
                 TreeNodeIndex* childOffsets,
                 TreeNodeIndex* parents)
{
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numInternalNodes; ++i)
    {
        TreeNodeIndex idxA    = leafToInternal[i];
        KeyType prefix        = prefixes[idxA];
        KeyType nodeKey       = decodePlaceholderBit(prefix);
        unsigned prefixLength = decodePrefixLength(prefix);
        unsigned level        = prefixLength / 3;
        assert(level < maxTreeLevel<KeyType>{});

        KeyType childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

        TreeNodeIndex leafSearchStart = levelRange[level + 1];
        TreeNodeIndex leafSearchEnd   = levelRange[level + 2];
        TreeNodeIndex childIdx =
            stl::lower_bound(prefixes + leafSearchStart, prefixes + leafSearchEnd, childPrefix) - prefixes;

        if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
        {
            childOffsets[idxA] = childIdx;
            // We only store the parent once for every group of 8 siblings.
            // This works as long as each node always has 8 siblings.
            // Subtract one because the root has no siblings.
            parents[(childIdx - 1) / 8] = idxA;
        }
    }
}

//! @brief determine the octree subdivision level boundaries
template<class KeyType>
void getLevelRangeCpu(const KeyType* nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex* levelRange)
{
    for (unsigned level = 0; level <= maxTreeLevel<KeyType>{}; ++level)
    {
        auto it = std::lower_bound(nodeKeys, nodeKeys + numNodes, encodePlaceholderBit(KeyType(0), 3 * level));
        levelRange[level] = TreeNodeIndex(it - nodeKeys);
    }
    levelRange[maxTreeLevel<KeyType>{} + 1] = numNodes;
}

/*! @brief construct the internal octree part of a given octree leaf cell array on the GPU
 *
 * @tparam       KeyType     unsigned 32- or 64-bit integer
 * @param[in]    cstoneTree  GPU buffer with the SFC leaf cell keys
 */
template<class KeyType>
void buildOctreeCpu(const KeyType* cstoneTree,
                    TreeNodeIndex numLeafNodes,
                    TreeNodeIndex numInternalNodes,
                    KeyType* prefixes,
                    TreeNodeIndex* childOffsets,
                    TreeNodeIndex* parents,
                    TreeNodeIndex* levelRange,
                    TreeNodeIndex* internalToLeaf,
                    TreeNodeIndex* leafToInternal)
{
    TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;
    createUnsortedLayoutCpu(cstoneTree, numInternalNodes, numLeafNodes, prefixes, internalToLeaf);
    sort_by_key(prefixes, prefixes + numNodes, internalToLeaf);

#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        leafToInternal[internalToLeaf[i]] = i;
    }
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        internalToLeaf[i] -= numInternalNodes;
    }
    getLevelRangeCpu(prefixes, numNodes, levelRange);

    std::fill(childOffsets, childOffsets + numNodes, 0);
    linkTreeCpu(prefixes, numInternalNodes, leafToInternal, levelRange, childOffsets, parents);
}

//! Octree data view, compatible with GPU data
template<class KeyType>
struct OctreeView
{
    using NodeType = std::conditional_t<std::is_const_v<KeyType>, const TreeNodeIndex, TreeNodeIndex>;
    TreeNodeIndex numLeafNodes;
    TreeNodeIndex numInternalNodes;
    TreeNodeIndex numNodes;

    KeyType* prefixes;
    NodeType* childOffsets;
    NodeType* parents;
    NodeType* levelRange;
    NodeType* internalToLeaf;
    NodeType* leafToInternal;
};

//! @brief combination of octree data needed for traversal with node properties
template<class T, class KeyType>
struct OctreeNsView
{
    //! @brief see OctreeData
    const KeyType* prefixes;
    const TreeNodeIndex* childOffsets;
    const TreeNodeIndex* internalToLeaf;
    const TreeNodeIndex* levelRange;

    //! @brief index of first particle for each node
    const LocalIndex* layout;
    //! @brief geometrical node centers and sizes
    const Vec3<T>* centers;
    const Vec3<T>* sizes;
};

template<class KeyType, class Accelerator>
class OctreeData
{
    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector =
        typename AccelSwitchType<Accelerator, std::vector, thrust::device_vector>::template type<ValueType>;

public:
    void resize(TreeNodeIndex numCsLeafNodes)
    {
        numLeafNodes     = numCsLeafNodes;
        numInternalNodes = (numLeafNodes - 1) / 7;
        numNodes         = numLeafNodes + numInternalNodes;

        prefixes.resize(numNodes);
        internalToLeaf.resize(numNodes);
        leafToInternal.resize(numNodes);
        childOffsets.resize(numNodes);
        // +1 to accommodate nodeOffsets in FocusedOctreeCore::update when numNodes == 1
        childOffsets.resize(numNodes + 1);

        TreeNodeIndex parentSize = std::max(1, (numNodes - 1) / 8);
        parents.resize(parentSize);

        //+1 due to level 0 and +1 due to the upper bound for the last level
        levelRange.resize(maxTreeLevel<KeyType>{} + 2);
    }

    OctreeView<KeyType> data()
    {
        return {numLeafNodes,       numInternalNodes,       numNodes,
                rawPtr(prefixes),   rawPtr(childOffsets),   rawPtr(parents),
                rawPtr(levelRange), rawPtr(internalToLeaf), rawPtr(leafToInternal)};
    }

    OctreeView<const KeyType> data() const
    {
        return {numLeafNodes,       numInternalNodes,       numNodes,
                rawPtr(prefixes),   rawPtr(childOffsets),   rawPtr(parents),
                rawPtr(levelRange), rawPtr(internalToLeaf), rawPtr(leafToInternal)};
    }

    TreeNodeIndex numNodes{0};
    TreeNodeIndex numLeafNodes{0};
    TreeNodeIndex numInternalNodes{0};

    //! @brief the SFC key and level of each node (Warren-Salmon placeholder-bit), length = numNodes
    AccVector<KeyType> prefixes;
    //! @brief the index of the first child of each node, a value of 0 indicates a leaf, length = numNodes
    AccVector<TreeNodeIndex> childOffsets;
    //! @brief stores the parent index for every group of 8 sibling nodes, length the (numNodes - 1) / 8
    AccVector<TreeNodeIndex> parents;
    //! @brief store the first node index of every tree level, length = maxTreeLevel + 2
    AccVector<TreeNodeIndex> levelRange;

    //! @brief maps internal to leaf (cstone) order
    AccVector<TreeNodeIndex> internalToLeaf;
    //! @brief maps leaf (cstone) order to internal level-sorted order
    AccVector<TreeNodeIndex> leafToInternal;
};

template<class KeyType>
void updateInternalTree(gsl::span<const KeyType> leaves, OctreeView<KeyType> o)
{
    assert(size_t(o.numLeafNodes) == nNodes(leaves));
    buildOctreeCpu(leaves.data(), o.numLeafNodes, o.numInternalNodes, o.prefixes, o.childOffsets, o.parents,
                   o.levelRange, o.internalToLeaf, o.leafToInternal);
}

} // namespace cstone

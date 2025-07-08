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
 * @brief Functions for finding peer ranks for point to point communication in global domains
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/macs.hpp"
#include "cstone/domain/domaindecomp.hpp"

namespace cstone
{

/*! @brief find peer ranks based on a multipole acceptance criterion and dual tree traversal
 *
 * @tparam T            float or double
 * @tparam KeyType      32- or 64-bit unsigned integer
 * @param myRank        find peers for the globally assigned SFC segment with index myRank
 * @param assignment    Decomposition of the global SFC into segments
 * @param domainTree    octree built on top of the global cornerstone leaves
 * @param box           global coordinate bounding box
 * @param invThetaEff   1/theta + s, effective inverse opening parameter
 * @return              list of segment indices (i.e. "ranks") that contain tree leaf nodes
 *                      that fail the MAC paired with at least one tree leaf node inside
 *                      the @p myRank segment. This list contains at least the segments
 *                      at the surface of the @p myRank segment and possibly additional
 *                      segments for low opening angles and/or low global resolution in
 *                      @p domainTree.
 *
 * Note: This function guarantees mutuality, if rank A identifies B as peer, then also
 *       rank B will have A as peer
 *
 * Except for @p myRank, this function acts on data that is identical on all MPI ranks and
 * doesn't need to do any communication.
 */
template<class T, template<class> class TreeType, class KeyType>
std::vector<int> findPeersMac(int myRank,
                              const SfcAssignment<KeyType>& assignment,
                              const TreeType<KeyType>& domainTree,
                              const Box<T>& box,
                              float invThetaEff)
{
    KeyType domainStart = assignment[myRank];
    KeyType domainEnd   = assignment[myRank + 1];

    int maxCoord   = 1u << maxTreeLevel<KeyType>{};
    float roundOff = 1 + 1e-6; // ensure that peers are picked up in case of a numerical tie
    auto ellipse   = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invThetaEff * roundOff;
    auto pbc_t     = BoundaryType::periodic;
    auto pbc       = Vec3<int>{box.boundaryX() == pbc_t, box.boundaryY() == pbc_t, box.boundaryZ() == pbc_t} * maxCoord;

    auto crossFocusPairs = [domainStart, domainEnd, ellipse, pbc, &tree = domainTree](TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aFocusOverlap = overlapTwoRanges(domainStart, domainEnd, tree.codeStart(a), tree.codeEnd(a));
        bool bInFocus      = containedIn(tree.codeStart(b), tree.codeEnd(b), domainStart, domainEnd);
        // node a has to overlap/be contained in the focus, while b must not be inside it
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = sfcIBox(sfcKey(tree.codeStart(a)), tree.level(a));
        IBox bBox = sfcIBox(sfcKey(tree.codeStart(b)), tree.level(b));
        return !minMacMutualInt(aBox, bBox, ellipse, pbc);
    };

    auto m2l = [](TreeNodeIndex, TreeNodeIndex) {};

    std::vector<int> peerRanks(assignment.numRanks(), 0);
    auto p2p = [&domainTree, &assignment, &peerRanks](TreeNodeIndex /*a*/, TreeNodeIndex b)
    {
        int peerRank = assignment.findRank(domainTree.codeStart(b));
        if (peerRanks[peerRank] == 0) { peerRanks[peerRank] = 1; }
    };

    std::vector<KeyType> spanningNodeKeys(spanSfcRange(domainStart, domainEnd) + 1);
    spanSfcRange(domainStart, domainEnd, spanningNodeKeys.data());
    spanningNodeKeys.back() = domainEnd;

    const KeyType* nodeKeys         = domainTree.nodeKeys().data();
    const TreeNodeIndex* levelRange = domainTree.levelRange().data();

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < spanningNodeKeys.size() - 1; ++i)
    {
        TreeNodeIndex nodeIdx = locateNode(spanningNodeKeys[i], spanningNodeKeys[i + 1], nodeKeys, levelRange);
        dualTraversal(domainTree, nodeIdx, 0, crossFocusPairs, m2l, p2p);
    }

    std::vector<int> ret;
    for (int i = 0; i < int(peerRanks.size()); ++i)
    {
        if (peerRanks[i]) { ret.push_back(i); }
    }

    return ret;
}

//! @brief Args identical to findPeersMac, but implemented with single tree traversal for comparison
template<class KeyType, class T>
std::vector<int> findPeersMacStt(int myRank,
                                 const SfcAssignment<KeyType>& assignment,
                                 const Octree<KeyType>& octree,
                                 const Box<T>& box,
                                 float invThetaEff)
{
    KeyType domainStart     = assignment[myRank];
    KeyType domainEnd       = assignment[myRank + 1];
    const KeyType* leaves   = octree.treeLeaves().data();
    TreeNodeIndex firstLeaf = findNodeAbove(leaves, octree.numLeafNodes(), domainStart);
    TreeNodeIndex lastLeaf  = findNodeAbove(leaves, octree.numLeafNodes(), domainEnd);

    int maxCoord = 1u << maxTreeLevel<KeyType>{};
    auto ellipse = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invThetaEff;
    auto pbc_t   = BoundaryType::periodic;
    auto pbc     = Vec3<int>{box.boundaryX() == pbc_t, box.boundaryY() == pbc_t, box.boundaryZ() == pbc_t} * maxCoord;

    std::vector<int> peers(assignment.numRanks());

#pragma omp parallel for
    for (TreeNodeIndex i = firstLeaf; i < lastLeaf; ++i)
    {
        IBox target = sfcIBox(sfcKey(leaves[i]), sfcKey(leaves[i + 1]));

        auto violatesMac = [target, ellipse, pbc, &octree, domainStart, domainEnd](TreeNodeIndex idx)
        {
            KeyType nodeStart = octree.codeStart(idx);
            KeyType nodeEnd   = octree.codeEnd(idx);
            // if the tree node with index idx is fully contained in the focus, we stop traversal
            if (containedIn(nodeStart, nodeEnd, domainStart, domainEnd)) { return false; }

            IBox source = sfcIBox(sfcKey(nodeStart), octree.level(idx));
            return !minMacMutualInt(target, source, ellipse, pbc);
        };

        auto markLeafIdx = [&octree, &peers, &assignment](TreeNodeIndex idx)
        {
            int peerRank    = assignment.findRank(octree.codeStart(idx));
            peers[peerRank] = 1;
        };

        singleTraversal(octree.childOffsets().data(), violatesMac, markLeafIdx);
    }

    std::vector<int> ret;
    for (int i = 0; i < int(peers.size()); ++i)
    {
        if (peers[i]) { ret.push_back(i); }
    }

    return ret;
}

} // namespace cstone

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
 * @brief Test functions used to find peer ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/peers.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

template<class KeyType>
void findMacPeers64grid(int rank, float theta, bool pbc, const std::vector<int>& reference)
{
    Box<double> box{-1, 1, pbc};
    Octree<KeyType> octree;
    octree.update(makeUniformNLevelTree<KeyType>(64, 1));

    SpaceCurveAssignment assignment(octree.numLeafNodes());
    for (int i = 0; i < octree.numLeafNodes(); ++i)
    {
        assignment.addRange(Rank(i), i, i + 1, 1);
    }

    std::vector<int> peers = findPeersMac(rank, assignment, octree, box, theta);
    EXPECT_EQ(peers, reference);
}

TEST(Peers, findMacGrid64)
{
    // just the surface
    findMacPeers64grid<unsigned>(0, 1.1, false, {1, 2, 3, 4, 5, 6, 7});
    findMacPeers64grid<uint64_t>(0, 1.1, false, {1, 2, 3, 4, 5, 6, 7});
}

TEST(Peers, findMacGrid64Narrow)
{
    findMacPeers64grid<unsigned>(0, 1.0, false, {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 17, 20, 21, 32, 33, 34, 35});
    findMacPeers64grid<uint64_t>(0, 1.0, false, {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 17, 20, 21, 32, 33, 34, 35});
}

TEST(Peers, findMacGrid64PBC)
{
    // just the surface + PBC
    findMacPeers64grid<unsigned>(
        0, 1.1, true, {1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18, 19, 22, 23, 27, 31, 36, 37, 38, 39, 45, 47, 54, 55, 63});
    findMacPeers64grid<uint64_t>(
        0, 1.1, true, {1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 18, 19, 22, 23, 27, 31, 36, 37, 38, 39, 45, 47, 54, 55, 63});
}

//! @brief reference peer search, all-all leaf comparison
template<class T, class KeyType>
std::vector<int> findPeersAll2All(int myRank, const SpaceCurveAssignment& assignment, gsl::span<const KeyType> tree,
                                  const Box<T>& box, float theta)
{
    TreeNodeIndex firstIdx = assignment.firstNodeIdx(myRank);
    TreeNodeIndex lastIdx = assignment.lastNodeIdx(myRank);
    float invThetaSq = 1.0f / (theta * theta);

    std::vector<IBox> boxes(nNodes(tree));
    for (TreeNodeIndex i = 0; i < nNodes(tree); ++i)
        boxes[i] = mortonIBox(tree[i], tree[i + 1]);

    std::vector<int> peers(assignment.numRanks());
    for (TreeNodeIndex i = firstIdx; i < lastIdx; ++i)
        for (TreeNodeIndex j = 0; j < nNodes(tree); ++j)
            if (!minDistanceMacMutual<KeyType>(boxes[i], boxes[j], box, invThetaSq)) peers[assignment.findRank(j)] = 1;

    std::vector<int> ret;
    for (int i = 0; i < peers.size(); ++i)
        if (peers[i] && i != myRank) { ret.push_back(i); }

    return ret;
}

template<class KeyType>
void findPeers()
{
    Box<double> box{-1, 1};
    int nParticles = 100000;
    int bucketSize = 64;
    int numRanks = 50;

    auto codes = makeRandomGaussianKeys<KeyType>(nParticles);

    Octree<KeyType> octree;
    auto [tree, counts] = computeOctree(codes.data(), codes.data() + nParticles, bucketSize);
    octree.update(tree.begin(), tree.end());

    SpaceCurveAssignment assignment = singleRangeSfcSplit(counts, numRanks);

    int probeRank = numRanks / 2;
    std::vector<int> peersDtt = findPeersMac(probeRank, assignment, octree, box, 0.5);
    std::vector<int> peersStt = findPeersMacStt(probeRank, assignment, octree, box, 0.5);
    std::vector<int> peersA2A = findPeersAll2All(probeRank, assignment, octree.treeLeaves(), box, 0.5);
    EXPECT_EQ(peersDtt, peersStt);
    EXPECT_EQ(peersDtt, peersA2A);

    // check for mutuality
    for (int peerRank : peersDtt)
    {
        std::vector<int> peersOfPeerDtt = findPeersMac(peerRank, assignment, octree, box, 0.5);

        // std::vector<int> peersOfPeerStt = findPeersMacStt(peerRank, assignment, octree, box, 0.5);
        // EXPECT_EQ(peersDtt, peersStt);
        std::vector<int> peersOfPeerA2A = findPeersAll2All(peerRank, assignment, octree.treeLeaves(), box, 0.5);
        EXPECT_EQ(peersOfPeerDtt, peersOfPeerA2A);

        // the peers of the peers of the probeRank have to have probeRank as peer
        EXPECT_TRUE(std::find(begin(peersOfPeerDtt), end(peersOfPeerDtt), probeRank) != end(peersOfPeerDtt));
    }
}

TEST(Peers, find)
{
    findPeers<unsigned>();
    findPeers<uint64_t>();
}

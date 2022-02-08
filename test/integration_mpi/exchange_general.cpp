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
 * @brief Test global octree build together with domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/peers.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

/*! @brief test for particle-count-exchange of distributed focused octrees
 *
 * First, all ranks create numRanks * numParticles random gaussian particle coordinates.
 * Since the random number generator is seeded with the same value on all ranks, all of them
 * will generate exactly the same numRanks * numParticles coordinates.
 *
 * This common coordinate set is then used to build a focus tree on each rank, using
 * non-communicating local algorithms to serve as the reference.
 * Each rank then takes the <thisRank>-th fraction of the common coordinate set and discards the other coordinates,
 * such that all ranks combined still have the original common set.
 * From the distributed coordinate set, the same focused trees are then built, but with distributed communicating
 * algorithms. This should yield the same tree on each rank as the local case,
 */
template<class KeyType, class T>
static void generalExchangeRandomGaussian(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 10.0;

    Box<T> box{-1, 1};

    /*******************************/
    /* identical data on all ranks */

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);

    auto [tree, counts] = computeOctree(coords.particleKeys().data(),
                                        coords.particleKeys().data() + coords.particleKeys().size(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

    auto assignment = singleRangeSfcSplit(counts, numRanks);

    /*******************************/

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, theta);

    KeyType focusStart = tree[assignment.firstNodeIdx(thisRank)];
    KeyType focusEnd   = tree[assignment.lastNodeIdx(thisRank)];

    // locate particles assigned to thisRank
    auto firstAssignedIndex = findNodeAbove<KeyType>(coords.particleKeys(), focusStart);
    auto lastAssignedIndex  = findNodeAbove<KeyType>(coords.particleKeys(), focusEnd);

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstAssignedIndex, coords.x().begin() + lastAssignedIndex);
    std::vector<T> y(coords.y().begin() + firstAssignedIndex, coords.y().begin() + lastAssignedIndex);
    std::vector<T> z(coords.z().begin() + firstAssignedIndex, coords.z().begin() + lastAssignedIndex);

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.
    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    FocusedOctree<KeyType> focusTree(thisRank, numRanks, bucketSizeLocal, theta);
    focusTree.converge(box, particleKeys, peers, assignment, tree, counts);

    const Octree<KeyType>& octree = focusTree.octree();
    std::vector<int> testCounts(octree.numTreeNodes(), -1);

    for (TreeNodeIndex i = 0; i < octree.numTreeNodes(); ++i)
    {
        KeyType nodeStart = octree.codeStart(i);
        KeyType nodeEnd   = octree.codeEnd(i);
        bool inFocus      = focusStart <= nodeStart && nodeEnd <= focusEnd;

        if (octree.isLeaf(i) && inFocus)
        {
            testCounts[i] =
                calculateNodeCount(nodeStart, nodeEnd, particleKeys.data(), particleKeys.data() + particleKeys.size(),
                                   std::numeric_limits<int>::max());
        }
    }

    auto sumFunction = [](auto a, auto b, auto c, auto d, auto e, auto f, auto g, auto h)
    { return a + b + c + d + e + f + g + h; };
    upsweep(octree, testCounts.data(), sumFunction);

    focusTree.template peerExchange<int>(peers, testCounts, static_cast<int>(P2pTags::focusPeerCounts) + 2);
    focusTree.template globalExchange<int>(domainTree, testCounts, sumFunction);
    upsweep(octree, testCounts.data(), sumFunction);

    {
        for (TreeNodeIndex i = 0; i < testCounts.size(); ++i)
        {
            KeyType nodeStart = octree.codeStart(i);
            KeyType nodeEnd   = octree.codeEnd(i);

            unsigned referenceCount = calculateNodeCount(nodeStart, nodeEnd, coords.particleKeys().data(),
                                                         coords.particleKeys().data() + coords.particleKeys().size(),
                                                         std::numeric_limits<int>::max());
            EXPECT_EQ(testCounts[i], referenceCount);
        }
    }

    EXPECT_EQ(testCounts[0], numRanks * numParticles);
}

TEST(GeneralFocusExchange, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeRandomGaussian<unsigned, double>(rank, nRanks);
    generalExchangeRandomGaussian<uint64_t, double>(rank, nRanks);
    generalExchangeRandomGaussian<unsigned, float>(rank, nRanks);
    generalExchangeRandomGaussian<uint64_t, float>(rank, nRanks);
}


template<class KeyType, class T>
static void generalExchangeSourceCenter(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 10.0;

    Box<T> box{-1, 1};

    /*******************************/
    /* identical data on all ranks */

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);
    std::vector<T> globalMasses(numRanks * numParticles, 1.0 / (numRanks * numParticles));

    auto [tree, counts] = computeOctree(coords.particleKeys().data(),
                                        coords.particleKeys().data() + coords.particleKeys().size(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

    auto assignment = singleRangeSfcSplit(counts, numRanks);

    /*******************************/

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, theta);

    KeyType focusStart = tree[assignment.firstNodeIdx(thisRank)];
    KeyType focusEnd   = tree[assignment.lastNodeIdx(thisRank)];

    // locate particles assigned to thisRank
    auto firstAssignedIndex = findNodeAbove<KeyType>(coords.particleKeys(), focusStart);
    auto lastAssignedIndex  = findNodeAbove<KeyType>(coords.particleKeys(), focusEnd);

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstAssignedIndex, coords.x().begin() + lastAssignedIndex);
    std::vector<T> y(coords.y().begin() + firstAssignedIndex, coords.y().begin() + lastAssignedIndex);
    std::vector<T> z(coords.z().begin() + firstAssignedIndex, coords.z().begin() + lastAssignedIndex);
    std::vector<T> m(globalMasses.begin() + firstAssignedIndex, globalMasses.begin() + lastAssignedIndex);

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.
    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    FocusedOctree<KeyType> focusTree(thisRank, numRanks, bucketSizeLocal, theta);
    focusTree.converge(box, particleKeys, peers, assignment, tree, counts);

    const Octree<KeyType>& octree = focusTree.octree();

    std::vector<SourceCenterType<T>> sourceCenter(octree.numTreeNodes());
    computeLeafMassCenter<T, T, T, KeyType>(x, y, z, m, particleKeys, octree, sourceCenter);
    upsweep(octree, sourceCenter.data(), CombineSourceCenter<T>{});

    focusTree.template peerExchange<SourceCenterType<T>>(peers, sourceCenter,
                                                         static_cast<int>(P2pTags::focusPeerCounts) + 2);
    focusTree.template globalExchange<SourceCenterType<T>>(domainTree, sourceCenter, CombineSourceCenter<T>{});
    upsweep(octree, sourceCenter.data(), CombineSourceCenter<T>{});

    constexpr T tol = std::is_same_v<T, double> ? 1e-10 : 1e-4;
    {
        for (TreeNodeIndex i = 0; i < sourceCenter.size(); ++i)
        {
            KeyType nodeStart = octree.codeStart(i);
            KeyType nodeEnd   = octree.codeEnd(i);

            LocalIndex startIndex = findNodeAbove<KeyType>(coords.particleKeys(), nodeStart);
            LocalIndex endIndex   = findNodeAbove<KeyType>(coords.particleKeys(), nodeEnd);

            SourceCenterType<T> reference =
                massCenter<T, T, T>(coords.x(), coords.y(), coords.z(), globalMasses, startIndex, endIndex);

            EXPECT_NEAR(sourceCenter[i][0], reference[0], tol);
            EXPECT_NEAR(sourceCenter[i][1], reference[1], tol);
            EXPECT_NEAR(sourceCenter[i][2], reference[2], tol);
            EXPECT_NEAR(sourceCenter[i][3], reference[3], tol);
        }
    }
    EXPECT_NEAR(sourceCenter[0][3], 1.0, tol);
}

TEST(GeneralFocusExchange, sourceCenter)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeSourceCenter<unsigned, double>(rank, nRanks);
    generalExchangeSourceCenter<uint64_t, double>(rank, nRanks);
    generalExchangeSourceCenter<unsigned, float>(rank, nRanks);
    generalExchangeSourceCenter<uint64_t, float>(rank, nRanks);
}
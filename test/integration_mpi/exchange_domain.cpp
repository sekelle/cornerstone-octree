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
 * @brief Tests the particle exchange used for exchanging assigned particles, i.e. not halos.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <gtest/gtest.h>

#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

/*! @brief all-to-all exchange, the most communication possible
 *
 * @tparam T         float, double or int
 * @param thisRank   executing rank
 * @param numRanks   total number of ranks
 *
 * Each rank keeps (1/numRanks)-th of its local elements and sends the
 * other numRanks-1 chunks to the other numRanks-1 ranks.
 */
template<class T>
void exchangeAllToAll(int thisRank, int numRanks)
{
    int gridSize = 64;

    std::vector<T> x(gridSize), y(gridSize);
    std::vector<LocalIndex> ordering(gridSize);

    std::iota(begin(x), end(x), 0);
    // unique element id across all ranks
    std::iota(begin(y), end(y), gridSize * thisRank);
    // start from trivial ordering
    std::iota(begin(ordering), end(ordering), 0);

    {
        // A simple, but nontrivial ordering.
        // Simulates the use case where the x,y,z coordinate arrays
        // are not sorted according to the Morton code ordering for which
        // the index ranges in the SendList are valid.
        int swap1 = 0;
        int swap2 = gridSize - 1;
        std::swap(x[swap1], x[swap2]);
        std::swap(y[swap1], y[swap2]);
        std::swap(ordering[swap1], ordering[swap2]);
    }

    int segmentSize = gridSize / numRanks;

    SendList sendList(numRanks);
    for (int rank = 0; rank < numRanks; ++rank)
    {
        int lower = rank * segmentSize;
        int upper = lower + segmentSize;

        if (rank == numRanks - 1) upper += gridSize % numRanks;

        sendList[rank].addRange(lower, upper);
    }

    // there's only one range per rank
    segmentSize              = sendList[thisRank].count(0);
    int numParticlesThisRank = segmentSize * numRanks;

    reallocate(std::max(numParticlesThisRank, int(x.size())), x, y);
    exchangeParticles(sendList, Rank(thisRank), 0, gridSize, x.size(), numParticlesThisRank, ordering.data(), x.data(),
                      y.data());
    reallocate(numParticlesThisRank, x, y);

    std::vector<T> refX(numParticlesThisRank);
    for (int rank = 0; rank < numRanks; ++rank)
    {
        std::iota(begin(refX) + rank * segmentSize, begin(refX) + rank * segmentSize + segmentSize,
                  sendList[thisRank].rangeStart(0));
    }

    std::vector<T> refY;
    for (int rank = 0; rank < numRanks; ++rank)
    {
        int seqStart = rank * gridSize + (gridSize / numRanks) * thisRank;

        for (int i = 0; i < segmentSize; ++i)
            refY.push_back(seqStart++);
    }

    std::sort(begin(x), end(x));
    std::sort(begin(refX), end(refX));

    // received particles are in indeterminate order
    std::sort(begin(y), end(y));

    EXPECT_EQ(refX, x);
    EXPECT_EQ(refY, y);
}

TEST(GlobalDomain, exchangeAllToAll)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    exchangeAllToAll<double>(rank, nRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeAllToAll<float>(rank, nRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeAllToAll<int>(rank, nRanks);
    MPI_Barrier(MPI_COMM_WORLD);
}

void exchangeCyclicNeighbors(int thisRank, int numRanks)
{
    int gridSize = 64;

    // x and y are filled with one value that is different for each rank
    std::vector<double> x(gridSize, thisRank);
    std::vector<float> y(gridSize, -thisRank);
    std::vector<util::array<int, 2>> testArray(gridSize, {thisRank, -thisRank});

    std::vector<LocalIndex> ordering(gridSize);
    std::iota(begin(ordering), end(ordering), 0);

    // send the last nex elements to the next rank
    int nex      = 10;
    int nextRank = (thisRank + 1) % numRanks;

    SendList sendList(numRanks);
    // keep all but the last nex elements
    sendList[thisRank].addRange(0, gridSize - nex);
    // send last nex to nextRank
    sendList[nextRank].addRange(gridSize - nex, gridSize);

    exchangeParticles(sendList, Rank(thisRank), 0, gridSize, gridSize, gridSize, ordering.data(), x.data(), y.data(),
                      testArray.data());

    int incomingRank = (thisRank - 1 + numRanks) % numRanks;
    std::vector<double> refX(gridSize, thisRank);
    std::fill(begin(refX) + gridSize - nex, end(refX), incomingRank);

    std::vector<float> refY(gridSize, -thisRank);
    std::fill(begin(refY) + gridSize - nex, end(refY), -incomingRank);

    std::vector<util::array<int, 2>> testArrayRef(gridSize, {thisRank, -thisRank});
    std::fill(begin(testArrayRef) + gridSize - nex, end(testArrayRef),
              util::array<int, 2>{incomingRank, -incomingRank});

    EXPECT_EQ(refX, x);
    EXPECT_EQ(refY, y);
    EXPECT_EQ(testArrayRef, testArray);
}

TEST(GlobalDomain, exchangeCyclicNeighbors)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    exchangeCyclicNeighbors(rank, numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
}

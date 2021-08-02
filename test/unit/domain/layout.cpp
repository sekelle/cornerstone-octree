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
 * @brief Test functions used to determine the arrangement of halo and assigned particles
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/layout.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

//! @brief tests extraction of SFC keys for all nodes marked as halos within an index range
TEST(Layout, extractMarkedElements)
{
    std::vector<unsigned> leaves{0,1,2,3,4,5,6,7,8,9,10};
    std::vector<int>   haloFlags{0,0,0,1,1,1,0,1,0,1};

    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 0);
        std::vector<unsigned> reference{};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 3);
        std::vector<unsigned> reference{};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 4);
        std::vector<unsigned> reference{3,4};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 5);
        std::vector<unsigned> reference{3,5};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 7);
        std::vector<unsigned> reference{3,6};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 0, 10);
        std::vector<unsigned> reference{3,6,7,8,9,10};
        EXPECT_EQ(reqKeys, reference);
    }
    {
        std::vector<unsigned> reqKeys = extractMarkedElements<unsigned>(leaves, haloFlags, 9, 10);
        std::vector<unsigned> reference{9,10};
        EXPECT_EQ(reqKeys, reference);
    }
}

TEST(Layout, computeHaloReceiveList)
{
    std::vector<LocalParticleIndex> layout{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> haloFlags{1, 0, 1, 1, 0, 0, 0, 1, 1, 0};

    std::vector<int> peers{0, 2};

    int numRanks = 3;
    std::vector<TreeIndexPair> assignment(numRanks);

    assignment[0] = TreeIndexPair(0, 4);
    assignment[1] = TreeIndexPair(4, 6);
    assignment[2] = TreeIndexPair(6, 10);

    SendList receiveList = computeHaloReceiveList(layout, haloFlags, assignment, peers);

    SendList reference(numRanks);
    reference[0].addRange(0, 1);
    reference[0].addRange(2, 4);
    reference[2].addRange(7, 9);

    EXPECT_EQ(receiveList, reference);
}

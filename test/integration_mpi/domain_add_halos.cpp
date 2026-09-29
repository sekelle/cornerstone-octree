/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test Domain::addHalos, halos requested by key on top of the distance-based halos
 */

#include <array>
#include <set>

#include <mpi.h>
#include <gtest/gtest.h>

#include "cstone/domain/domain.hpp"

using namespace cstone;

/*! @brief particles on a regular n[0] x n[1] x n[2] grid, each with an id property
 *
 * The smoothing lengths are much smaller than the grid spacing, so sync() leaves out many grid neighbours of the
 * assigned particles. Requesting all 26 neighbours by key must make them all present, as a mesh needs for the
 * elements that share a node. The z-spacing is @p zStretch times the x,y-spacing, like a layer of tall elements.
 */
template<class KeyType, class T>
void addNeighbourHalos(int rank, int numRanks, std::array<int, 3> n, T zStretch)
{
    const int N = n[0] * n[1] * n[2];
    const T dx  = T(1) / (n[0] - 1);
    const T dz  = zStretch * dx;

    auto gridIndex = [](T c, T spacing) { return int(std::lround(c / spacing)); };
    auto gridId    = [n](int i, int j, int k) { return T((i * n[1] + j) * n[2] + k); };

    // hand out the grid round-robin, so the initial distribution has nothing to do with the SFC
    std::vector<T> x, y, z, h, id;
    for (int i = rank; i < N; i += numRanks)
    {
        x.push_back((i / (n[1] * n[2])) * dx);
        y.push_back(((i / n[2]) % n[1]) * dx);
        z.push_back((i % n[2]) * dz);
        h.push_back(0.05 * dx);
        id.push_back(i);
    }

    Box<T> box(0, 1, 0, (n[1] - 1) * dx, 0, (n[2] - 1) * dz);
    Domain<KeyType, T> domain(execution::cpu, rank, numRanks, 16, 8, 1.0, MPI_COMM_WORLD, box);
    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tie(id), std::tie(s1, s2, s3));

    std::vector<T> assignedBefore(x.begin() + domain.startIndex(), x.begin() + domain.endIndex());
    std::vector<T> idBefore(id.begin() + domain.startIndex(), id.begin() + domain.endIndex());
    std::set<T> presentBefore;
    for (size_t i = 0; i < x.size(); ++i)
    {
        presentBefore.insert(gridId(gridIndex(x[i], dx), gridIndex(y[i], dx), gridIndex(z[i], dz)));
    }

    std::vector<KeyType> request;
    std::set<T> neighbours;
    for (size_t p = domain.startIndex(); p < domain.endIndex(); ++p)
    {
        int i = gridIndex(x[p], dx), j = gridIndex(y[p], dx), k = gridIndex(z[p], dz);
        for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
                for (int dk = -1; dk <= 1; ++dk)
                {
                    int a = i + di, b = j + dj, c = k + dk;
                    if (a < 0 || b < 0 || c < 0 || a >= n[0] || b >= n[1] || c >= n[2]) { continue; }
                    request.push_back(sfc3D<SfcKind<KeyType>>(a * dx, b * dx, c * dz, domain.box()));
                    neighbours.insert(gridId(a, b, c));
                }
    }

    // make sure the test exercises addHalos: sync alone must have left out some neighbours
    int missingBefore = 0;
    for (T i : neighbours)
    {
        missingBefore += !presentBefore.count(i);
    }
    MPI_Allreduce(MPI_IN_PLACE, &missingBefore, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (numRanks > 1) { EXPECT_GT(missingBefore, 0); }

    domain.addHalos(request, keys, x, y, z, h, std::tie(id), std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(id), s1, s2);

    ASSERT_EQ(x.size(), domain.nParticlesWithHalos());
    ASSERT_EQ(id.size(), domain.nParticlesWithHalos());

    std::vector<T> assignedAfter(x.begin() + domain.startIndex(), x.begin() + domain.endIndex());
    std::vector<T> idAfter(id.begin() + domain.startIndex(), id.begin() + domain.endIndex());
    EXPECT_EQ(assignedBefore, assignedAfter);
    EXPECT_EQ(idBefore, idAfter);

    std::set<T> presentAfter;
    for (size_t i = 0; i < x.size(); ++i)
    {
        // halo properties arrive through the new exchange pattern and must match the halo coordinates
        EXPECT_EQ(id[i], gridId(gridIndex(x[i], dx), gridIndex(y[i], dx), gridIndex(z[i], dz)));
        presentAfter.insert(id[i]);
    }
    for (T i : presentBefore)
    {
        EXPECT_TRUE(presentAfter.count(i)) << "lost particle " << i;
    }
    for (T i : neighbours)
    {
        EXPECT_TRUE(presentAfter.count(i)) << "missing neighbour " << i;
    }

    std::vector<KeyType> keysCheck(x.size());
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(keysCheck.data()), x.size(), domain.box());
    EXPECT_EQ(keys, keysCheck);
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));

    // the domain stays usable: the next sync starts from the extended layout
    domain.sync(keys, x, y, z, h, std::tie(id), std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(id), s1, s2);
    for (size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_EQ(id[i], gridId(gridIndex(x[i], dx), gridIndex(y[i], dx), gridIndex(z[i], dz)));
    }
}

TEST(DomainAddHalos, neighbours)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    addNeighbourHalos<uint64_t, double>(rank, numRanks, {16, 16, 16}, 1.0);
    addNeighbourHalos<unsigned, float>(rank, numRanks, {16, 16, 16}, 1.0);
}

/*! @brief two layers of particles far apart compared to their in-layer spacing
 *
 * A requested neighbour in the other layer can lie in a focus cell that passes the MAC and spans several ranks.
 * addHalos has to refine the focus tree until that cell belongs to one rank.
 */
TEST(DomainAddHalos, tallLayers)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    addNeighbourHalos<uint64_t, double>(rank, numRanks, {64, 64, 2}, 150.0);
    addNeighbourHalos<unsigned, float>(rank, numRanks, {64, 64, 2}, 150.0f);
}

//! @brief no rank requests anything: the halos of the previous sync must stay unchanged
TEST(DomainAddHalos, noKeys)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using T       = double;
    using KeyType = uint64_t;
    constexpr int n = 16;
    std::vector<T> x, y, z, h;
    for (int i = rank; i < n * n * n; i += numRanks)
    {
        x.push_back((i / (n * n)) / T(n - 1));
        y.push_back(((i / n) % n) / T(n - 1));
        z.push_back((i % n) / T(n - 1));
        h.push_back(0.3 / (n - 1));
    }

    Domain<KeyType, T> domain(execution::cpu, rank, numRanks, 16, 8, 1.0, MPI_COMM_WORLD, Box<T>{0, 1});
    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    auto xBefore = x;
    auto start   = domain.startIndex();
    domain.addHalos({}, keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));
    EXPECT_EQ(domain.startIndex(), start);
    EXPECT_EQ(x, xBefore);
}

// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <limits>

#include <mpi.h>
#include <gtest/gtest.h>

#include "cstone/domain/domain.hpp"

using namespace cstone;

template<class KeyType>
void sharedRepresentativeHalos(int rank, int numRanks)
{
    constexpr int n           = 4;
    constexpr int cells[5][4] = {{1, 2, 4, 7}, {0, 1, 2, 4}, {3, 1, 2, 7}, {5, 1, 4, 7}, {6, 2, 4, 7}};
    Box<double> box(-.05, 1.05);
    std::vector<std::array<double, 3>> representatives;
    std::vector<double> x, y, z, h;
    std::vector<KeyType> id;

    // Several tets share each representative, forcing some partition boundaries to the finest SFC levels.
    for (int iz = 0; iz < n; ++iz)
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < n; ++ix)
                for (const auto& cell : cells)
                {
                    std::array<KeyType, 4> keys;
                    std::array<std::array<double, 3>, 4> points;
                    for (int c = 0; c < 4; ++c)
                    {
                        int corner = cell[c] ^ ((ix + iy + iz) & 1);
                        points[c]  = {double(ix + (corner & 1)) / n, double(iy + ((corner >> 1) & 1)) / n,
                                      double(iz + ((corner >> 2) & 1)) / n};
                        keys[c]    = sfc3D<SfcKind<KeyType>>(points[c][0], points[c][1], points[c][2], box);
                    }
                    int rep         = std::min_element(keys.begin(), keys.end()) - keys.begin();
                    KeyType element = representatives.size();
                    representatives.push_back(points[rep]);
                    if (element % numRanks != KeyType(rank)) { continue; }
                    x.push_back(points[rep][0]);
                    y.push_back(points[rep][1]);
                    z.push_back(points[rep][2]);
                    id.push_back(element);
                    double shortest = 1;
                    for (int a = 0; a < 4; ++a)
                        for (int b = a + 1; b < 4; ++b)
                        {
                            double distance = 0;
                            for (int d = 0; d < 3; ++d)
                                distance += (points[a][d] - points[b][d]) * (points[a][d] - points[b][d]);
                            shortest = std::min(shortest, std::sqrt(distance));
                        }
                    h.push_back(shortest);
                }

    Domain<KeyType, double> domain(execution::cpu, rank, numRanks, 8, 1, .5, MPI_COMM_WORLD, box);
    std::vector<KeyType> keys(x.size()), ordering;
    std::vector<double> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tie(id), std::tie(s1, s2, s3, ordering));

    int invalid = 0;
    for (size_t i = 0; i < id.size(); ++i)
    {
        bool owned    = i >= domain.startIndex() && i < domain.endIndex();
        int suppliers = owned;
        for (const auto& range : domain.incomingHaloIndices())
            suppliers += i >= size_t(range.start()) && i < size_t(range.end());
        invalid += suppliers != 1;
        // Old buffer contents must not hide a missing receive.
        if (!owned) { id[i] = std::numeric_limits<KeyType>::max(); }
    }
    domain.exchangeHalos(std::tie(id), s1, s2);
    for (size_t i = 0; i < id.size(); ++i)
    {
        if (id[i] >= representatives.size()) { ++invalid; }
        else
        {
            const auto& p = representatives[id[i]];
            invalid += x[i] != p[0] || y[i] != p[1] || z[i] != p[2];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(invalid, 0) << sizeof(KeyType) * 8 << "-bit layout has missing suppliers or invalid halo properties";
}

TEST(DomainHaloDepth, sharedRepresentatives)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    ASSERT_EQ(numRanks, 4);
    sharedRepresentativeHalos<uint32_t>(rank, numRanks);
    sharedRepresentativeHalos<uint64_t>(rank, numRanks);
}

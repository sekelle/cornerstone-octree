#include <gtest/gtest.h>

#include "coord_samples/random.hpp"

using namespace cstone;

TEST(MixedHilbertSample, Long1DDomain)
{
    using real        = double;
    using IntegerType = unsigned;
    int n             = 10;

    Box<real> box{0, 100, -1, 1, -1, 1};
    RandomCoordinates<real, Sfc1DMixedKind<IntegerType>> c(n, box);

    std::vector<IntegerType> testCodes(n);
    computeSfc1D3DKeys(c.x().data(), c.y().data(), c.z().data(), Sfc1DMixedKindPointer(testCodes.data()), n, box);

    EXPECT_TRUE(std::is_sorted(testCodes.begin(), testCodes.end()));
    for (int i{0}; i < n; ++i) {
        std::cout << "Coord:\t" << c.x()[i] << "\t" << c.y()[i] << "\t" << c.z()[i] << "\tCode binary:\t" << std::bitset<32>(testCodes[i]) << std::endl;
    }
}

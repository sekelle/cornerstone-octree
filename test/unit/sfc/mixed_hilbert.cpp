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

    unsigned levels_1D = 2;

    std::vector<IntegerType> testCodes(n);
    computeSfc1D3DKeys(c.x().data(), c.y().data(), c.z().data(), Sfc1DMixedKindPointer(testCodes.data()), n, box,
                       levels_1D);

    EXPECT_TRUE(std::is_sorted(testCodes.begin(), testCodes.end()));
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest1D3D()
{
    int numKeys      = 10;
    int maxCoordLong = (1 << maxTreeLevel<KeyType>{}) - 1;
    unsigned levels_2D{2};
    int maxCoordShort = (1 << (maxTreeLevel<KeyType>{} - levels_2D)) - 1;

    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution_long(0, maxCoordLong);
    std::uniform_int_distribution<unsigned> distribution_short(0, maxCoordShort);

    auto getRandLong  = [&distribution_long, &gen]() { return distribution_long(gen); };
    auto getRandShort = [&distribution_short, &gen]() { return distribution_short(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRandLong);
    std::generate(begin(y), end(y), getRandShort);
    std::generate(begin(z), end(z), getRandShort);

    for (int i = 0; i < numKeys; ++i)
    {
        KeyType hilbertKey = iHilbert1DMixed<KeyType>(x[i], y[i], z[i], levels_2D, 0);

        auto [a, b, c] = decodeHilbert1DMixed(hilbertKey, levels_2D, 0);
        EXPECT_EQ(x[i], a);
        EXPECT_EQ(y[i], b);
        EXPECT_EQ(z[i], c);
    }
}

TEST(MixedHilbert1D3D, InversionTest2D3D)
{
    inversionTest1D3D<unsigned>();
    inversionTest1D3D<uint64_t>();
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest2D3D()
{
    int numKeys      = 10;
    int maxCoordLong = (1 << maxTreeLevel<KeyType>{}) - 1;
    unsigned levels_2D{2};
    int maxCoordShort = (1 << (maxTreeLevel<KeyType>{} - levels_2D)) - 1;

    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution_long(0, maxCoordLong);
    std::uniform_int_distribution<unsigned> distribution_short(0, maxCoordShort);

    auto getRandLong  = [&distribution_long, &gen]() { return distribution_long(gen); };
    auto getRandShort = [&distribution_short, &gen]() { return distribution_short(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRandShort);
    std::generate(begin(y), end(y), getRandLong);
    std::generate(begin(z), end(z), getRandLong);

    for (int i = 0; i < numKeys; ++i)
    {
        KeyType hilbertKey = iHilbert2DMixed<KeyType>(x[i], y[i], z[i], levels_2D, 0);

        auto [a, b, c] = decodeHilbert2DMixed(hilbertKey, levels_2D, 0);
        EXPECT_EQ(x[i], a);
        EXPECT_EQ(y[i], b);
        EXPECT_EQ(z[i], c);
    }
}

TEST(MixedHilbert2D3D, InversionTest2D3D)
{
    inversionTest2D3D<unsigned>();
    inversionTest2D3D<uint64_t>();
}

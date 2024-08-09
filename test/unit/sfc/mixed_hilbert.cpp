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
    for (int i{0}; i < n; ++i)
    {
        std::cout << "Coord:\t" << c.x()[i] << "\t" << c.y()[i] << "\t" << c.z()[i] << "\tCode binary:\t"
                  << std::bitset<32>(testCodes[i]) << std::endl;
    }
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest2D3D()
{
    int numKeys      = 3;
    int maxCoordLong = (1 << maxTreeLevel<KeyType>{}) - 1;
    int levels_2D{2};
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
        std::cout << "x : " << std::bitset<10>(x[i]) << " y : " << std::bitset<10>(y[i])
                  << " z : " << std::bitset<10>(z[i]) << std::endl;
        std::cout << "a : " << std::bitset<10>(a) << " b : " << std::bitset<10>(b) << " c : " << std::bitset<10>(c)
                  << std::endl;
        std::cout << "hilbert  key: " << std::bitset<32>(hilbertKey) << std::endl
                  << "original key: " << std::bitset<32>(iHilbert<KeyType>(x[i], y[i], z[i])) << std::endl;
        EXPECT_EQ(x[i], a);
        EXPECT_EQ(y[i], b);
        EXPECT_EQ(z[i], c);
    }
}

TEST(MixedHilbertSample, InversionTest2D3D)
{
    inversionTest2D3D<unsigned>();
    // inversionTest2D3D<uint64_t>();
}
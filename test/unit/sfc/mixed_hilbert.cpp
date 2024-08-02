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
void inversionTest()
{
    int numKeys  = 3;
    int maxCoord = (1 << maxTreeLevel<KeyType>{}) - 1;

    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution(0, maxCoord);

    auto getRand = [&distribution, &gen]() { return distribution(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    for (int i = 0; i < numKeys; ++i)
    {
        KeyType hilbertKey = iHilbert1DMixed<KeyType>(x[i], y[i], z[i], 2, 0);

        auto [a, b, c] = decodeHilbert(hilbertKey);
        std::cout << "x: " << x[i] << " y: " << y[i] << " z: " << z[i] << " hilbertKey: " << std::bitset<32>(hilbertKey)
                  << " a: " << a << " b: " << b << " c: " << c
                  << " original key: " << std::bitset<32>(iHilbert<KeyType>(x[i], y[i], z[i])) << std::endl;
        // EXPECT_EQ(x[i], a);
        // EXPECT_EQ(y[i], b);
        // EXPECT_EQ(z[i], c);
    }
}

TEST(MixedHilbertCode, inversion)
{
    inversionTest<unsigned>();
    inversionTest<uint64_t>();
}

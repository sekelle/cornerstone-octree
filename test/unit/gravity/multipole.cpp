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

#include <iomanip>

#include "gtest/gtest.h"

#include "cstone/gravity/multipole.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

//! @brief Tests direct particle-to-particle gravity interactions
TEST(GravityKernel, P2P)
{
    using T = double;

    T eps2 = 0.05 * 0.05;

    // target
    T x = 1;
    T y = 1;
    T z = 1;

    // source
    T xs[2] = {2, -2};
    T ys[2] = {2, -2};
    T zs[2] = {2, -2};
    T m[2]  = {1, 1};

    T xacc = 0;
    T yacc = 0;
    T zacc = 0;

    particle2particle(x, y, z, xs, ys, zs, m, 2, eps2, &xacc, &yacc, &zacc);

    EXPECT_DOUBLE_EQ(xacc, 0.17082940372214045);
    EXPECT_DOUBLE_EQ(yacc, 0.17082940372214045);
    EXPECT_DOUBLE_EQ(zacc, 0.17082940372214045);
}


//! @brief Tests the gravity interaction of a multipole with a particle
TEST(GravityKernel, M2P)
{
    using T = double;

    Box<T> box(-1, 1);
    LocalParticleIndex numParticles = 100;
    T eps2 = 0.05 * 0.05;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    GravityMultipole<T> multipole = particle2Multipole(x, y, z, masses.data(), numParticles, 0.5, 0.5, 0.5);

    std::array<T, 3> target    = {-8, 0, 0};
    std::array<T, 3> accDirect = {0, 0, 0};
    std::array<T, 3> accApprox = {0, 0, 0};

    particle2particle(target[0], target[1], target[2], x, y, z, masses.data(), numParticles, eps2,
                      &accDirect[0], &accDirect[1], &accDirect[2]);

    std::cout << std::fixed;
    std::cout << "direct: " << accDirect[0] << " " << accDirect[1] << " " << accDirect[2] << std::endl;

    multipole2particle(target[0], target[1], target[2], multipole, eps2, &accApprox[0], &accApprox[1], &accApprox[2]);

    std::cout << "approx: " << accApprox[0] << " " << accApprox[1] << " " << accApprox[2] << std::endl;
}
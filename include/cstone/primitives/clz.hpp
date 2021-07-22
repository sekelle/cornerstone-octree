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
 * @brief count leading zeros in unsigned integers
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"

namespace detail
{

//! @brief count leading zeros, does not handle an input of 0
constexpr int clz32(uint32_t x)
{
    constexpr int debruijn32[32] = {0, 31, 9, 30, 3, 8,  13, 29, 2,  5,  7,  21, 12, 24, 28, 19,
                                    1, 10, 4, 14, 6, 22, 25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    x++;
    return debruijn32[x * 0x076be629 >> 27u];
}

//! @brief count leading zeros, does not handle an input of 0
constexpr int clz64(uint64_t x)
{
    constexpr int debruijn64[64] = {0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61, 54, 58, 35, 52, 50, 42,
                                    21, 44, 38, 32, 29, 23, 17, 11, 4,  62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43,
                                    31, 22, 10, 45, 25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};

    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    x |= x >> 32u;

    return 63 - debruijn64[x * 0x03f79d71b4cb0a89ul >> 58u];
}

} // namespace detail

/*! @brief count leading zeros, for 32 and 64 bit integers,
 *         return the number of bits in the input type for an input value of 0
 *
 * @tparam I  32- or 64-bit unsigned integer type
 * @param x   input number
 * @return    number of leading zeros, or the number of bits in the input type
 *            for an input value of 0
 */
HOST_DEVICE_FUN
constexpr int countLeadingZeros(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __clz(x);
    // with GCC and clang, we can use the builtin implementation
    // this also works with the intel compiler, which also defines __GNUC__
#elif defined(__GNUC__) || defined(__clang__)

    // if the target architecture is Haswell or later,
    // __builtin_clz(l) is implemented with the LZCNT instruction
    // which returns the number of bits for an input of zero,
    // so this check is not required in that case (flag: -march=haswell)
    if (x == 0) return 8 * sizeof(uint32_t);
    return __builtin_clz(x);

#else
    if (x == 0) return 8 * sizeof(uint32_t);
    return detail::clz32(x);

#endif
}

HOST_DEVICE_FUN
constexpr int countLeadingZeros(uint64_t x)
{
#ifdef __CUDA_ARCH__
    return __clzll(x);
    // with GCC and clang, we can use the builtin implementation
    // this also works with the intel compiler, which also defines __GNUC__
#elif defined(__GNUC__) || defined(__clang__)

    // if the target architecture is Haswell or later,
    // __builtin_clz(l) is implemented with the LZCNT instruction
    // which returns the number of bits for an input of zero,
    // so this check is not required in that case (flag: -march=haswell)
    if (x == 0) return 8 * sizeof(uint64_t);
    return __builtin_clzl(x);

#else
    if (x == 0) return 8 * sizeof(uint64_t);
    return detail::clz64(x);
#endif
}


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

#include "annotation.hpp"

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
#endif
}

//! @brief returns number of trailing zero-bits, does not handle an input of zero
HOST_DEVICE_FUN
constexpr int countTrailingZeros(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __ffs(x) - 1;
#else
    return __builtin_ctz(x);
#endif
}

HOST_DEVICE_FUN
constexpr int countTrailingZeros(uint64_t x)
{
#ifdef __CUDA_ARCH__
    return __ffsll(x) - 1;
#else
    return __builtin_ctzl(x);
#endif
}

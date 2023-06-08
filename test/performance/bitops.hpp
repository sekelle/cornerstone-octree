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
 * @brief  Common operations on SFC keys that do not depend on the specific SFC used
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "annotation.hpp"
#include "definitions.h"

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

namespace cstone
{

/*! @brief compute the maximum range of an octree node at a given subdivision level
 *
 * @tparam KeyType    32- or 64-bit unsigned integer type
 * @param  treeLevel  octree subdivision level
 * @return            the range
 *
 * At treeLevel 0, the range is the entire 30 or 63 bits used in the SFC code.
 * After that, the range decreases by 3 bits for each level.
 *
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType nodeRange(unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<KeyType>{});
    unsigned shifts = maxTreeLevel<KeyType>{} - treeLevel;

    return KeyType(1ul << (3u * shifts));
}

//! @brief compute ceil(log8(n))
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned log8ceil(KeyType n)
{
    if (n == 0) { return 0; }

    unsigned lz = countLeadingZeros(n - 1);
    return maxTreeLevel<KeyType>{} - (lz - unusedBits<KeyType>{}) / 3;
}

//! @brief check whether n is a power of 8
template<class KeyType>
HOST_DEVICE_FUN constexpr bool isPowerOf8(KeyType n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits<KeyType>{};
    return lz % 3 == 0 && !(n & (n - 1));
}

/*! @brief calculate common prefix (cpr) of two SFC keys
 *
 * @tparam KeyType  32 or 64 bit unsigned integer
 * @param  key1     first SFC code key
 * @param  key2     second SFC code key
 * @return          number of continuous identical bits, counting from MSB
 *                  minus the 2 unused bits in 32 bit codes or minus the 1 unused bit
 *                  in 64 bit codes.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr int commonPrefix(KeyType key1, KeyType key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<KeyType>{};
}

/*! @brief return octree subdivision level corresponding to codeRange
 *
 * @tparam KeyType   32- or 64-bit unsigned integer type
 * @param codeRange  input SFC code range
 * @return           octree subdivision level 0-10 (32-bit) or 0-21 (64-bit)
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned treeLevel(KeyType codeRange)
{
    assert(isPowerOf8(codeRange));
    return (countLeadingZeros(codeRange - 1) - unusedBits<KeyType>{}) / 3;
}

/*! @brief convert a plain SFC key into the placeholder bit format (Warren-Salmon 1993)
 *
 * @tparam KeyType         32- or 64-bit unsigned integer
 * @param code             input SFC key
 * @param prefixLength     number of leading bits which are part of the code
 * @return                 code shifted by trailing zeros and prepended with 1-bit
 *
 * Example: encodePlaceholderBit(06350000000, 9) -> 01635 (in octal)
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType encodePlaceholderBit(KeyType code, int prefixLength)
{
    int nShifts             = 3 * maxTreeLevel<KeyType>{} - prefixLength;
    KeyType ret             = code >> nShifts;
    KeyType placeHolderMask = KeyType(1) << prefixLength;

    return placeHolderMask | ret;
}

//! @brief returns the number of key-bits in the input @p code
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned decodePrefixLength(KeyType code)
{
    return 8 * sizeof(KeyType) - 1 - countLeadingZeros(code);
}

/*! @brief decode an SFC key in Warren-Salmon placeholder bit format
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param code       input SFC key with 1-bit prepended
 * @return           SFC-key without 1-bit and shifted to most significant bit
 *
 * Inverts encodePlaceholderBit.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType decodePlaceholderBit(KeyType code)
{
    int prefixLength        = decodePrefixLength(code);
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    KeyType ret             = code ^ placeHolderMask;

    return ret << (3 * maxTreeLevel<KeyType>{} - prefixLength);
}

/*! @brief extract the n-th octal digit from an SFC key, starting from the most significant
 *
 * @tparam KeyType   32- or 64-bit unsigned integer type
 * @param code       Input SFC key code
 * @param position   Which digit place to extract. Return values will be meaningful for
 *                   @p position in [1:11] for 32-bit keys and in [1:22] for 64-bit keys and
 *                   will be zero otherwise, but a value of 0 for @p position can also be specified
 *                   to detect whether the 31st or 63rd bit for the last cornerstone is non-zero.
 *                   (The last cornerstone has a value of nodeRange<KeyType>(0) = 2^31 or 2^63)
 * @return           The value of the digit at place @p position
 *
 * The position argument correspondence to octal digit places has been chosen such that
 * octalDigit(code, pos) returns the octant at octree division level pos.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned octalDigit(KeyType code, unsigned position)
{
    return (code >> (3u * (maxTreeLevel<KeyType>{} - position))) & 7u;
}

//! @brief cut down the input SFC code to the start code of the enclosing box at <treeLevel>
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType enclosingBoxCode(KeyType key, unsigned treeLevel)
{
    KeyType mask = KeyType(nodeRange<KeyType>(treeLevel) - 1);

    return KeyType(key & ~mask);
}

} // namespace cstone

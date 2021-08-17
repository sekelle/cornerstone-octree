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
 * @brief (halo-)box overlap functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"

namespace cstone
{

//! @brief standard criterion for two ranges a-b and c-d to overlap, a<b and c<d
template<class T>
HOST_DEVICE_FUN constexpr bool overlapTwoRanges(T a, T b, T c, T d)
{
    assert(a<=b && c<=d);
    return b > c && d > a;
}

/*! @brief determine whether two ranges ab and cd overlap
 *
 * @tparam R  periodic range
 * @return    true or false
 *
 * Some restrictions apply, no input value can be further than R
 * from the periodic range.
 */
template<int R>
HOST_DEVICE_FUN constexpr bool overlapRange(int a, int b, int c, int d)
{
    assert(a >= -R);
    assert(a < R);
    assert(b > 0);
    assert(b <= 2*R);

    assert(c >= -R);
    assert(c < R);
    assert(d > 0);
    assert(d <= 2*R);

    return overlapTwoRanges(a,b,c,d) ||
           overlapTwoRanges(a+R, b+R, c, d) ||
           overlapTwoRanges(a, b, c+R, d+R);
}

//! @brief check whether two boxes overlap. takes PBC into account, boxes can wrap around
template<class KeyType>
HOST_DEVICE_FUN inline bool overlap(const IBox& a, const IBox& b)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};

    bool xOverlap = overlapRange<maxCoord>(a.xmin(), a.xmax(), b.xmin(), b.xmax());
    bool yOverlap = overlapRange<maxCoord>(a.ymin(), a.ymax(), b.ymin(), b.ymax());
    bool zOverlap = overlapRange<maxCoord>(a.zmin(), a.zmax(), b.zmin(), b.zmax());

    return xOverlap && yOverlap && zOverlap;
}

/*! @brief Check whether a coordinate box is fully contained in a Morton code range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param codeStart  Morton code range start
 * @param codeEnd    Morton code range end
 * @param box        3D box with x,y,z integer coordinates in [0,2^maxTreeLevel<KeyType>{}-1]
 * @return           true if the box is fully contained within the specified Morton code range
 */
template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType codeStart, KeyType codeEnd, const IBox& box)
{
    // volume 0 boxes are not possible if makeHaloBox was used to generate it
    assert(box.xmin() < box.xmax());
    assert(box.ymin() < box.ymax());
    assert(box.zmin() < box.zmax());

    constexpr unsigned pbcRange = 1u << maxTreeLevel<KeyType>{};
    if (stl::min(stl::min(box.xmin(), box.ymin()), box.zmin()) < 0 ||
        stl::max(stl::max(box.xmax(), box.ymax()), box.zmax()) > pbcRange)
    {
        // any box that wraps around a PBC boundary cannot be contained within
        // any octree node, except the full root node
        return codeStart == 0 && codeEnd == nodeRange<KeyType>(0);
    }

    KeyType lowCode  = iSfcKey<SfcKind<KeyType>>(box.xmin(), box.ymin(), box.zmin());
    KeyType highCode = iSfcKey<SfcKind<KeyType>>(box.xmax() - 1, box.ymax() - 1, box.zmax() - 1);
    auto envelope    = smallestCommonBox(lowCode, highCode);

    return (util::get<0>(envelope) >= codeStart) && (util::get<1>(envelope) <= codeEnd);
}

/*! @brief determine whether a binary/octree node (prefix, prefixLength) is fully contained in an SFC range
 *
 * @tparam KeyType       32- or 64-bit unsigned integer
 * @param  prefix        lowest SFC code of the tree node
 * @param  prefixLength  range of the tree node in bits,
 *                       corresponding SFC range is 2^(3*maxTreeLevel<KeyType>{} - prefixLength)
 * @param  codeStart     start of the SFC range
 * @param  codeEnd       end of the SFC range
 * @return
 */
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType nodeStart, KeyType nodeEnd, KeyType codeStart, KeyType codeEnd)
{
    return !(nodeStart < codeStart || nodeEnd > codeEnd);
}

template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType prefixBitKey, KeyType codeStart, KeyType codeEnd)
{
    unsigned prefixLength = decodePrefixLength(prefixBitKey);
    KeyType firstPrefix   = decodePlaceholderBit(prefixBitKey);
    KeyType secondPrefix  = firstPrefix + (KeyType(1) << (3*maxTreeLevel<KeyType>{} - prefixLength));
    return !(firstPrefix < codeStart || secondPrefix > codeEnd);
}

template<class KeyType>
HOST_DEVICE_FUN inline int addDelta(int value, int delta, bool pbc)
{
    constexpr int maxCoordinate = (1u << maxTreeLevel<KeyType>{});

    int temp = value + delta;
    if (pbc) return temp;
    else     return stl::min(stl::max(0, temp), maxCoordinate);
}

//! @brief create a box with specified radius around node delineated by codeStart/End
template<class KeyType, class CoordinateType, class RadiusType>
HOST_DEVICE_FUN IBox makeHaloBox(const IBox& nodeBox, RadiusType radius, const Box<CoordinateType>& box)
{
    int dx = toNBitIntCeil<KeyType>(radius * box.ilx());
    int dy = toNBitIntCeil<KeyType>(radius * box.ily());
    int dz = toNBitIntCeil<KeyType>(radius * box.ilz());

    return IBox(addDelta<KeyType>(nodeBox.xmin(), -dx, box.pbcX()), addDelta<KeyType>(nodeBox.xmax(), dx, box.pbcX()),
                addDelta<KeyType>(nodeBox.ymin(), -dy, box.pbcY()), addDelta<KeyType>(nodeBox.ymax(), dy, box.pbcY()),
                addDelta<KeyType>(nodeBox.zmin(), -dz, box.pbcZ()), addDelta<KeyType>(nodeBox.zmax(), dz, box.pbcZ()));
}

//! @brief create a box with specified radius around node delineated by codeStart/End
template<class KeyType, class CoordinateType, class RadiusType>
HOST_DEVICE_FUN IBox makeHaloBox(KeyType codeStart, KeyType codeEnd, RadiusType radius, const Box<CoordinateType>& box)
{
    // disallow boxes with no volume
    assert(codeEnd > codeStart);
    IBox nodeBox = sfcIBox(sfcKey(codeStart), sfcKey(codeEnd));
    return makeHaloBox<KeyType>(nodeBox, radius, box);
}

} // namespace cstone


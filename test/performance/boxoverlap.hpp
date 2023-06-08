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

#include "box.hpp"
#include "sfc.hpp"

namespace cstone
{

//! @brief returns true if the cuboid defined by center and size is contained within the bounding box
template<class T>
HOST_DEVICE_FUN bool insideBox(const Vec3<T>& center, const Vec3<T>& size, const Box<T>& box)
{
    Vec3<T> globalMin{box.xmin(), box.ymin(), box.zmin()};
    Vec3<T> globalMax{box.xmax(), box.ymax(), box.zmax()};
    Vec3<T> boxMin = center - size;
    Vec3<T> boxMax = center + size;
    return boxMin[0] >= globalMin[0] && boxMin[1] >= globalMin[1] && boxMin[2] >= globalMin[2] &&
           boxMax[0] <= globalMax[0] && boxMax[1] <= globalMax[1] && boxMax[2] <= globalMax[2];
}

//! @brief returns the smallest distance vector of point X to box b, 0 if X is in b
template<class T>
HOST_DEVICE_FUN Vec3<T> minDistance(const Vec3<T>& X, const Vec3<T>& bCenter, const Vec3<T>& bSize)
{
    Vec3<T> dX = abs(bCenter - X) - bSize;
    dX += abs(dX);
    dX *= T(0.5);
    return dX;
}

//! @brief returns the smallest periodic distance vector of point X to box b, 0 if X is in b
template<class T>
HOST_DEVICE_FUN Vec3<T> minDistance(const Vec3<T>& X, const Vec3<T>& bCenter, const Vec3<T>& bSize, const Box<T>& box)
{
    Vec3<T> dX = bCenter - X;
    dX         = abs(applyPbc(dX, box));
    dX -= bSize;
    dX += abs(dX);
    dX *= T(0.5);

    return dX;
}

//! @brief returns the smallest distance vector between two boxes, 0 if they overlap
template<class T>
HOST_DEVICE_FUN Vec3<T>
minDistance(const Vec3<T>& aCenter, const Vec3<T>& aSize, const Vec3<T>& bCenter, const Vec3<T>& bSize)
{
    Vec3<T> dX = abs(bCenter - aCenter) - aSize - bSize;
    dX += abs(dX);
    dX *= T(0.5);

    return dX;
}

//! @brief returns the smallest periodic distance vector between two boxes, 0 if they overlap
template<class T>
HOST_DEVICE_FUN Vec3<T> minDistance(
    const Vec3<T>& aCenter, const Vec3<T>& aSize, const Vec3<T>& bCenter, const Vec3<T>& bSize, const Box<T>& box)
{
    Vec3<T> dX = bCenter - aCenter;
    dX         = abs(applyPbc(dX, box));
    dX -= aSize;
    dX -= bSize;
    dX += abs(dX);
    dX *= T(0.5);

    return dX;
}

//! @brief Convenience wrapper to minDistance. This should only be used for testing.
template<class KeyType, class T>
HOST_DEVICE_FUN T minDistanceSq(IBox a, IBox b, const Box<T>& box)
{
    auto [aCenter, aSize] = centerAndSize<KeyType>(a, box);
    auto [bCenter, bSize] = centerAndSize<KeyType>(b, box);
    return norm2(minDistance(aCenter, aSize, bCenter, bSize, box));
}

} // namespace cstone

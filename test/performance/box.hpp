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
 * @brief implements a bounding bounding box for floating point coordinates and integer indices
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cmath>

#include "annotation.hpp"
#include "stl.hpp"
#include "definitions.h"
#include "tuple.hpp"

namespace cstone
{

enum class BoundaryType : char
{
    open     = 0,
    periodic = 1,
    fixed    = 2
};

//! @brief stores the coordinate bounds
template<class T>
class Box
{

public:
    HOST_DEVICE_FUN constexpr Box(T xyzMin, T xyzMax, BoundaryType b = BoundaryType::open)
        : limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax}
        , lengths_{xyzMax - xyzMin, xyzMax - xyzMin, xyzMax - xyzMin}
        , inverseLengths_{T(1.) / (xyzMax - xyzMin), T(1.) / (xyzMax - xyzMin), T(1.) / (xyzMax - xyzMin)}
        , boundaries{b, b, b}
    {
    }

    HOST_DEVICE_FUN constexpr Box(T xmin,
                                  T xmax,
                                  T ymin,
                                  T ymax,
                                  T zmin,
                                  T zmax,
                                  BoundaryType bx = BoundaryType::open,
                                  BoundaryType by = BoundaryType::open,
                                  BoundaryType bz = BoundaryType::open)
        : limits{xmin, xmax, ymin, ymax, zmin, zmax}
        , lengths_{xmax - xmin, ymax - ymin, zmax - zmin}
        , inverseLengths_{T(1.) / (xmax - xmin), T(1.) / (ymax - ymin), T(1.) / (zmax - zmin)}
        , boundaries{bx, by, bz}
    {
    }

    HOST_DEVICE_FUN constexpr T xmin() const { return limits[0]; }
    HOST_DEVICE_FUN constexpr T xmax() const { return limits[1]; }
    HOST_DEVICE_FUN constexpr T ymin() const { return limits[2]; }
    HOST_DEVICE_FUN constexpr T ymax() const { return limits[3]; }
    HOST_DEVICE_FUN constexpr T zmin() const { return limits[4]; }
    HOST_DEVICE_FUN constexpr T zmax() const { return limits[5]; }

    //! @brief return edge lengths
    HOST_DEVICE_FUN constexpr T lx() const { return lengths_[0]; }
    HOST_DEVICE_FUN constexpr T ly() const { return lengths_[1]; }
    HOST_DEVICE_FUN constexpr T lz() const { return lengths_[2]; }

    //! @brief return inverse edge lengths
    HOST_DEVICE_FUN constexpr T ilx() const { return inverseLengths_[0]; }
    HOST_DEVICE_FUN constexpr T ily() const { return inverseLengths_[1]; }
    HOST_DEVICE_FUN constexpr T ilz() const { return inverseLengths_[2]; }

    HOST_DEVICE_FUN constexpr BoundaryType boundaryX() const { return boundaries[0]; } // NOLINT
    HOST_DEVICE_FUN constexpr BoundaryType boundaryY() const { return boundaries[1]; } // NOLINT
    HOST_DEVICE_FUN constexpr BoundaryType boundaryZ() const { return boundaries[2]; } // NOLINT

private:
    HOST_DEVICE_FUN
    friend constexpr bool operator==(const Box<T>& a, const Box<T>& b)
    {
        return a.limits[0] == b.limits[0] && a.limits[1] == b.limits[1] && a.limits[2] == b.limits[2] &&
               a.limits[3] == b.limits[3] && a.limits[4] == b.limits[4] && a.limits[5] == b.limits[5] &&
               a.boundaries[0] == b.boundaries[0] && a.boundaries[1] == b.boundaries[1] &&
               a.boundaries[2] == b.boundaries[2];
    }

    T limits[6];
    T lengths_[3];
    T inverseLengths_[3];
    BoundaryType boundaries[3];
};

//! @brief Compute the shortest periodic distance dX = A - B between two points,
template<class T>
HOST_DEVICE_FUN inline Vec3<T> applyPbc(Vec3<T> dX, const Box<T>& box)
{
    bool pbcX = (box.boundaryX() == BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == BoundaryType::periodic);

    dX[0] -= pbcX * box.lx() * std::rint(dX[0] * box.ilx());
    dX[1] -= pbcY * box.ly() * std::rint(dX[1] * box.ily());
    dX[2] -= pbcZ * box.lz() * std::rint(dX[2] * box.ilz());

    return dX;
}

/*! @brief stores octree index integer bounds
 */
template<class T>
class SimpleBox
{
public:
    HOST_DEVICE_FUN constexpr SimpleBox()
        : limits{0, 0, 0, 0, 0, 0}
    {
    }

    HOST_DEVICE_FUN constexpr SimpleBox(T xyzMin, T xyzMax)
        : limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax}
    {
    }

    HOST_DEVICE_FUN constexpr SimpleBox(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
        : limits{xmin, xmax, ymin, ymax, zmin, zmax}
    {
    }

    HOST_DEVICE_FUN constexpr T xmin() const { return limits[0]; } // NOLINT
    HOST_DEVICE_FUN constexpr T xmax() const { return limits[1]; } // NOLINT
    HOST_DEVICE_FUN constexpr T ymin() const { return limits[2]; } // NOLINT
    HOST_DEVICE_FUN constexpr T ymax() const { return limits[3]; } // NOLINT
    HOST_DEVICE_FUN constexpr T zmin() const { return limits[4]; } // NOLINT
    HOST_DEVICE_FUN constexpr T zmax() const { return limits[5]; } // NOLINT

    //! @brief return the shortest coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T minExtent() const // NOLINT
    {
        return stl::min(stl::min(xmax() - xmin(), ymax() - ymin()), zmax() - zmin());
    }

private:
    HOST_DEVICE_FUN
    friend constexpr bool operator==(const SimpleBox& a, const SimpleBox& b)
    {
        return a.limits[0] == b.limits[0] && a.limits[1] == b.limits[1] && a.limits[2] == b.limits[2] &&
               a.limits[3] == b.limits[3] && a.limits[4] == b.limits[4] && a.limits[5] == b.limits[5];
    }

    HOST_DEVICE_FUN
    friend bool operator<(const SimpleBox& a, const SimpleBox& b)
    {
        return util::tie(a.limits[0], a.limits[1], a.limits[2], a.limits[3], a.limits[4], a.limits[5]) <
               util::tie(b.limits[0], b.limits[1], b.limits[2], b.limits[3], b.limits[4], b.limits[5]);
    }

    T limits[6];
};

using IBox = SimpleBox<int>;

/*! @brief calculate floating point 3D center and radius of a and integer box and bounding box pair
 *
 * @tparam T         float or double
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param ibox       integer coordinate box
 * @param box        floating point bounding box
 * @return           the geometrical center and the vector from the center to the box corner farthest from the origin
 */
template<class KeyType, class T>
constexpr HOST_DEVICE_FUN util::tuple<Vec3<T>, Vec3<T>> centerAndSize(const IBox& ibox, const Box<T>& box)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / maxCoord;

    T halfUnitLengthX = T(0.5) * uL * box.lx();
    T halfUnitLengthY = T(0.5) * uL * box.ly();
    T halfUnitLengthZ = T(0.5) * uL * box.lz();
    Vec3<T> boxCenter = {box.xmin() + (ibox.xmax() + ibox.xmin()) * halfUnitLengthX,
                         box.ymin() + (ibox.ymax() + ibox.ymin()) * halfUnitLengthY,
                         box.zmin() + (ibox.zmax() + ibox.zmin()) * halfUnitLengthZ};
    Vec3<T> boxSize   = {(ibox.xmax() - ibox.xmin()) * halfUnitLengthX, (ibox.ymax() - ibox.ymin()) * halfUnitLengthY,
                         (ibox.zmax() - ibox.zmin()) * halfUnitLengthZ};

    return {boxCenter, boxSize};
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

} // namespace cstone

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
 * @brief Evaluate multipole acceptance criteria (MAC) on octree nodes
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include <vector>

#include "boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone
{

//! @brief compute 1/theta + s for the minimum distance MAC
HOST_DEVICE_FUN inline float invThetaMinMac(float theta) { return 1.0f / theta + 0.5f; }

//! @brief compute 1/theta + sqrt(3) for the worst-case vector MAC
HOST_DEVICE_FUN inline float invThetaVecMac(float theta) { return 1.0f / theta + std::sqrt(3.0f); }

/*! @brief Compute square of the acceptance radius for the minimum distance MAC
 *
 * @param prefix       SFC key of the tree cell with Warren-Salmon placeholder-bit
 * @param invThetaEff  1/theta + s (effective opening parameter)
 * @param box          global coordinate bounding box
 * @return             geometric center in the first 3 elements, the square of the distance from @p sourceCenter
 *                     beyond which the MAC fails or passes in the 4th element
 */
template<class T, class KeyType>
HOST_DEVICE_FUN Vec4<T> computeMinMacR2(KeyType prefix, float invThetaEff, const Box<T>& box)
{
    KeyType nodeKey  = decodePlaceholderBit(prefix);
    int prefixLength = decodePrefixLength(prefix);

    IBox cellBox              = sfcIBox(sfcKey(nodeKey), prefixLength / 3);
    auto [geoCenter, geoSize] = centerAndSize<KeyType>(cellBox, box);

    T l   = T(2) * max(geoSize);
    T mac = l * invThetaEff;

    return {geoCenter[0], geoCenter[1], geoCenter[2], mac * mac};
}

/*! @brief Compute square of the acceptance radius for the vector MAC
 *
 * @param prefix       SFC key of the tree cell with Warren-Salmon placeholder-bit
 * @param expCenter    expansion (com) center of the source (cell)
 * @param invTheta     1/theta (opening parameter)
 * @param box          global coordinate bounding box
 * @return             the square of the distance from @p sourceCenter beyond which the MAC fails or passes
 */
template<class T, class KeyType>
HOST_DEVICE_FUN T computeVecMacR2(KeyType prefix, Vec3<T> expCenter, float invTheta, const Box<T>& box)
{
    KeyType nodeKey  = decodePlaceholderBit(prefix);
    int prefixLength = decodePrefixLength(prefix);

    IBox cellBox              = sfcIBox(sfcKey(nodeKey), prefixLength / 3);
    auto [geoCenter, geoSize] = centerAndSize<KeyType>(cellBox, box);

    Vec3<T> dX = expCenter - geoCenter;

    T s   = sqrt(norm2(dX));
    T l   = T(2.0) * max(geoSize);
    T mac = l * invTheta + s;

    return mac * mac;
}

/*! @brief evaluate an arbitrary MAC with respect to a given target
 *
 * @tparam T             float or double
 * @param sourceCenter   expansion center of the MAC
 * @param macSq          squared acceptance radius around @p sourceCenter
 * @param targetCenter   target coordinate
 * @param targetSize     target half box length (>0) in all dimensions
 * @return                true if the target is closer to @p sourceCenter than the acceptance radius
 */
template<class T>
HOST_DEVICE_FUN bool evaluateMac(Vec3<T> sourceCenter, T macSq, Vec3<T> targetCenter, Vec3<T> targetSize)
{
    Vec3<T> dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= T(0.5);
    T R2 = norm2(dX);
    return R2 < std::abs(macSq);
}

/*! @brief evaluate an arbitrary MAC with respect to a given target
 *
 * @tparam T              float or double
 * @param  sourceCenter   source cell expansion center, can be geometric or center-mass, depending on
 *                        choice of MAC used to compute @p macSq
 * @param  macSq          squared multipole acceptance radius of the source cell
 * @param  targetCenter   geometric target cell center coordinates
 * @param  targetSize     geometric size of the target cell
 * @param  box            global coordinate bounding box
 * @return                true if the target is closer to @p sourceCenter than the acceptance radius
 */
template<class T>
HOST_DEVICE_FUN bool
evaluateMacPbc(Vec3<T> sourceCenter, T macSq, Vec3<T> targetCenter, Vec3<T> targetSize, const Box<T>& box)
{
    Vec3<T> dX = targetCenter - sourceCenter;

    dX = abs(applyPbc(dX, box));
    dX -= targetSize;
    dX += abs(dX);
    dX *= T(0.5);
    T R2 = norm2(dX);
    return R2 < std::abs(macSq);
}

//! @brief commutative version of the min-distance mac, based on floating point math
template<class T>
HOST_DEVICE_FUN bool minMacMutual(const Vec3<T>& centerA,
                                  const Vec3<T>& sizeA,
                                  const Vec3<T>& centerB,
                                  const Vec3<T>& sizeB,
                                  const Box<T>& box,
                                  float invTheta)
{
    Vec3<T> dX = minDistance(centerA, sizeA, centerB, sizeB, box);

    T distSq = norm2(dX);
    T sizeAB = 2 * stl::max(max(sizeA), max(sizeB));

    T mac = sizeAB * invTheta;

    return distSq > (mac * mac);
}

/*! @brief commutative combination of min-distance and vector map
 *
 * @param invThetaEff  1/theta + s, effective inverse opening parameter
 *
 * This MAC doesn't pass any A-B pairs that would fail either the min-distance
 * or vector MAC. Can be used instead of the vector mac when the mass center locations
 * are not known.
 */
template<class T>
HOST_DEVICE_FUN bool minVecMacMutual(const Vec3<T>& centerA,
                                     const Vec3<T>& sizeA,
                                     const Vec3<T>& centerB,
                                     const Vec3<T>& sizeB,
                                     const Box<T>& box,
                                     float invThetaEff)
{
    bool passA;
    {
        // A = target, B = source
        Vec3<T> dX = minDistance(centerB, centerA, sizeA, box);
        T mac      = max(sizeB) * 2 * invThetaEff;
        passA      = norm2(dX) > (mac * mac);
    }
    bool passB;
    {
        // B = target, A = source
        Vec3<T> dX = minDistance(centerA, centerB, sizeB, box);
        T mac      = max(sizeA) * 2 * invThetaEff;
        passB      = norm2(dX) > (mac * mac);
    }
    return passA && passB;
}

//! @brief mark all nodes of @p octree (leaves and internal) that fail the evaluateMac w.r.t to @p target
template<class T, class KeyType>
HOST_DEVICE_FUN void markMacPerBox(const Vec3<T>& targetCenter,
                                   const Vec3<T>& targetSize,
                                   const KeyType* prefixes,
                                   const TreeNodeIndex* childOffsets,
                                   const Vec4<T>* centers,
                                   const Box<T>& box,
                                   KeyType focusStart,
                                   KeyType focusEnd,
                                   char* markings)
{
    auto checkAndMarkMac =
        [&targetCenter, &targetSize, prefixes, &box, focusStart, focusEnd, centers, markings](TreeNodeIndex idx)
    {
        KeyType nodePrefix = prefixes[idx];
        KeyType nodeStart  = decodePlaceholderBit(nodePrefix);
        KeyType nodeEnd    = nodeStart + nodeRange<KeyType>(decodePrefixLength(nodePrefix) / 3);
        // if the tree node with index idx is fully contained in the focus, we stop traversal
        if (containedIn(nodeStart, nodeEnd, focusStart, focusEnd)) { return false; }

        Vec4<T> center   = centers[idx];
        bool violatesMac = evaluateMacPbc(makeVec3(center), center[3], targetCenter, targetSize, box);
        if (violatesMac && !markings[idx]) { markings[idx] = 1; }

        return violatesMac;
    };

    singleTraversal(childOffsets, checkAndMarkMac, [](TreeNodeIndex) {});
}

/*! @brief Mark each node in an octree that fails the MAC paired with any node from a given focus SFC range
 *
 * @tparam     T            float or double
 * @tparam     KeyType      32- or 64-bit unsigned integer
 * @param[in]  octree       octree, including internal part
 * @param[in]  centers      tree cell expansion (com) center coordinates and mac radius, size @p octree.numTreeNodes()
 * @param[in]  box          global coordinate bounding box
 * @param[in]  focusStart   lower SFC focus code
 * @param[in]  focusEnd     upper SFC focus code
 * @param[out] markings     array of length @p octree.numTreeNodes(), each position i
 *                          will be set to 1, if the node of @p octree with index i fails the MAC paired with
 *                          any node contained in the focus range [focusStart:focusEnd]
 */
template<class T, class KeyType>
void markMacs(const OctreeView<KeyType>& octree,
              const Vec4<T>* centers,
              const Box<T>& box,
              KeyType focusStart,
              KeyType focusEnd,
              char* markings)

{
    std::fill(markings, markings + octree.numInternalNodes + octree.numLeafNodes, 0);

    // find the minimum possible number of octree node boxes to cover the entire focus
    TreeNodeIndex numFocusBoxes = spanSfcRange(focusStart, focusEnd);
    std::vector<KeyType> focusCodes(numFocusBoxes + 1);
    spanSfcRange(focusStart, focusEnd, focusCodes.data());
    focusCodes.back() = focusEnd;

#pragma omp parallel for schedule(dynamic)
    for (TreeNodeIndex i = 0; i < numFocusBoxes; ++i)
    {
        IBox target                     = sfcIBox(sfcKey(focusCodes[i]), sfcKey(focusCodes[i + 1]));
        auto [targetCenter, targetSize] = centerAndSize<KeyType>(target, box);
        markMacPerBox(targetCenter, targetSize, octree.prefixes, octree.childOffsets, centers, box, focusStart,
                      focusEnd, markings);
    }
}

} // namespace cstone

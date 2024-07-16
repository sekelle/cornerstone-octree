/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Compute cell mass centers for use in focus tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/math.hpp"
#include "source_center.hpp"
#include "source_center_gpu.h"

namespace cstone
{

template<class Tc, class Tm, class Tf>
__global__ void computeLeafSourceCenterKernel(const Tc* x,
                                              const Tc* y,
                                              const Tc* z,
                                              const Tm* m,
                                              const TreeNodeIndex* leafToInternal,
                                              TreeNodeIndex numLeaves,
                                              const LocalIndex* layout,
                                              Vec4<Tf>* centers)
{
    TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeaves) { return; }

    TreeNodeIndex nodeIdx = leafToInternal[leafIdx];
    centers[nodeIdx]      = massCenter<Tf>(x, y, z, m, layout[leafIdx], layout[leafIdx + 1]);
}

template<class Tc, class Tm, class Tf>
void computeLeafSourceCenterGpu(const Tc* x,
                                const Tc* y,
                                const Tc* z,
                                const Tm* m,
                                const TreeNodeIndex* leafToInternal,
                                TreeNodeIndex numLeaves,
                                const LocalIndex* layout,
                                Vec4<Tf>* centers)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numLeaves, numThreads);

    computeLeafSourceCenterKernel<<<numBlocks, numThreads>>>(x, y, z, m, leafToInternal, numLeaves, layout, centers);
}

#define COMPUTE_LEAF_SOURCE_CENTER_GPU(Tc, Tm, Tf)                                                                     \
    template void computeLeafSourceCenterGpu(const Tc*, const Tc*, const Tc*, const Tm*, const TreeNodeIndex*,         \
                                             TreeNodeIndex, const LocalIndex*, Vec4<Tf>*);

COMPUTE_LEAF_SOURCE_CENTER_GPU(double, double, double);
COMPUTE_LEAF_SOURCE_CENTER_GPU(double, float, double);
COMPUTE_LEAF_SOURCE_CENTER_GPU(float, float, float);

template<class T>
__global__ void upsweepCentersKernel(TreeNodeIndex firstCell,
                                     TreeNodeIndex lastCell,
                                     const TreeNodeIndex* childOffsets,
                                     SourceCenterType<T>* centers)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    if (firstChild) { centers[cellIdx] = CombineSourceCenter<T>{}(cellIdx, firstChild, centers); }
}

template<class T>
void upsweepCentersGpu(int numLevels,
                       const TreeNodeIndex* levelRange,
                       const TreeNodeIndex* childOffsets,
                       SourceCenterType<T>* centers)
{
    constexpr int numThreads = 256;

    for (int level = numLevels - 1; level >= 0; level--)
    {
        int numCellsLevel = levelRange[level + 1] - levelRange[level];
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        if (numCellsLevel)
        {
            upsweepCentersKernel<<<numBlocks, numThreads>>>(levelRange[level], levelRange[level + 1], childOffsets,
                                                            centers);
        }
    }
}

template void upsweepCentersGpu(int, const TreeNodeIndex*, const TreeNodeIndex*, SourceCenterType<float>*);
template void upsweepCentersGpu(int, const TreeNodeIndex*, const TreeNodeIndex*, SourceCenterType<double>*);

template<class KeyType, class T>
__global__ void computeGeoCentersKernel(
    const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T> box)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }

    KeyType prefix                  = prefixes[i];
    KeyType startKey                = decodePlaceholderBit(prefix);
    unsigned level                  = decodePrefixLength(prefix) / 3;
    auto nodeBox                    = sfcIBox(sfcKey(startKey), level);
    util::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
}

template<class KeyType, class T>
void computeGeoCentersGpu(
    const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    computeGeoCentersKernel<<<numBlocks, numThreads>>>(prefixes, numNodes, centers, sizes, box);
}

#define GEO_CENTERS_GPU(KeyType, T)                                                                                    \
    template void computeGeoCentersGpu(const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers,              \
                                       Vec3<T>* sizes, const Box<T>& box)
GEO_CENTERS_GPU(uint32_t, float);
GEO_CENTERS_GPU(uint32_t, double);
GEO_CENTERS_GPU(uint64_t, float);
GEO_CENTERS_GPU(uint64_t, double);

template<class KeyType, class T>
__global__ void geoMacSpheresKernel(
    const KeyType* prefixes, TreeNodeIndex numNodes, SourceCenterType<T>* centers, float invTheta, Box<T> box)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    centers[i] = computeMinMacR2(prefixes[i], invTheta, box);
}

//! @brief set @p centers to geometric node centers with Mac radius l * invTheta
template<class KeyType, class T>
void geoMacSpheresGpu(
    const KeyType* prefixes, TreeNodeIndex numNodes, SourceCenterType<T>* centers, float invTheta, const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    geoMacSpheresKernel<<<numBlocks, numThreads>>>(prefixes, numNodes, centers, invTheta, box);
}

#define GEO_MAC_SPHERES_GPU(KeyType, T)                                                                                \
    template void geoMacSpheresGpu(const KeyType* prefixes, TreeNodeIndex numNodes, SourceCenterType<T>* centers,      \
                                   float invTheta, const Box<T>& box)
GEO_MAC_SPHERES_GPU(uint32_t, float);
GEO_MAC_SPHERES_GPU(uint32_t, double);
GEO_MAC_SPHERES_GPU(uint64_t, float);
GEO_MAC_SPHERES_GPU(uint64_t, double);

template<class KeyType, class T>
__global__ void
setMacKernel(const KeyType* prefixes, TreeNodeIndex numNodes, Vec4<T>* macSpheres, float invTheta, const Box<T> box)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }

    Vec4<T> center   = macSpheres[i];
    T mac            = computeVecMacR2(prefixes[i], util::makeVec3(center), invTheta, box);
    macSpheres[i][3] = (center[3] != T(0)) ? mac : T(0);
}

template<class KeyType, class T>
void setMacGpu(const KeyType* prefixes, TreeNodeIndex numNodes, Vec4<T>* macSpheres, float invTheta, const Box<T>& box)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    setMacKernel<<<numBlocks, numThreads>>>(prefixes, numNodes, macSpheres, invTheta, box);
}

#define SET_MAC_GPU(KeyType, T)                                                                                        \
    template void setMacGpu(const KeyType* prefixes, TreeNodeIndex numNodes, Vec4<T>* macSpheres, float invTheta,      \
                            const Box<T>& box)

SET_MAC_GPU(uint32_t, float);
SET_MAC_GPU(uint64_t, float);
SET_MAC_GPU(uint32_t, double);
SET_MAC_GPU(uint64_t, double);

template<class T>
__global__ void moveCentersKernel(const Vec3<T>* src, TreeNodeIndex numNodes, Vec4<T>* dest)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    dest[i][0] = src[i][0];
    dest[i][1] = src[i][1];
    dest[i][2] = src[i][2];
    dest[i][3] = 1.0;
}

template<class T>
void moveCenters(const Vec3<T>* src, TreeNodeIndex numNodes, Vec4<T>* dest)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = iceil(numNodes, numThreads);
    moveCentersKernel<<<numBlocks, numThreads>>>(src, numNodes, dest);
}

template void moveCenters(const Vec3<double>*, TreeNodeIndex, Vec4<double>*);
template void moveCenters(const Vec3<float>*, TreeNodeIndex, Vec4<float>*);

} // namespace cstone

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
 * @brief Generic octree traversal methods
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Single and dual tree traversal methods are the base algorithms for implementing
 * MAC evaluations, collision and surface detection etc.
 */

#pragma once

#include "annotation.hpp"

namespace cstone
{

template<class C, class A>
HOST_DEVICE_FUN void singleTraversal(const TreeNodeIndex* childOffsets, C&& continuationCriterion, A&& endpointAction)
{
    bool descend = continuationCriterion(0);
    if (!descend) return;

    if (childOffsets[0] == 0)
    {
        // root node is already the endpoint
        endpointAction(0);
        return;
    }

    TreeNodeIndex stack[128];
    stack[0] = 0;

    TreeNodeIndex stackPos = 1;
    TreeNodeIndex node     = 0; // start at the root

    do
    {
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = childOffsets[node] + octant;
            bool descend        = continuationCriterion(child);
            if (descend)
            {
                if (childOffsets[child] == 0)
                {
                    // endpoint reached with child is a leaf node
                    endpointAction(child);
                }
                else
                {
                    assert(stackPos < 128);
                    stack[stackPos++] = child; // push
                }
            }
        }
        node = stack[--stackPos];

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

} // namespace cstone

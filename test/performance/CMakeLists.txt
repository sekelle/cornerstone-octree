include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf)

# only scan.cpp provides some coverage beyond the unit tests
cstone_add_test(scan_perf EXECUTABLE scan_perf RANKS 1)

if(CMAKE_CUDA_COMPILER)
    cstone_add_cuda_performance_test(neighbor_driver.cu neighbors_test_gpu)
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu)

    set_source_files_properties(octree.cu PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)
endif()

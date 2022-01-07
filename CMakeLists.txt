
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

include_directories("/usr/local/cuda-10.2/samples/common/inc/")

add_library(surface STATIC
  surface_reconstruction.cpp
  voxel_structure.cu
  )

# Request that surface be built with -std=c++11
# As this is a public compile feature anything that links to surface
# will also build with -std=c++11
target_compile_features(surface PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( surface
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

if(BUILD_TESTING)

  add_executable(surface_test main_voxel_reconstruction.cu)

  set_target_properties(surface_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(surface_test PRIVATE surface)

  add_test(NAME surface_reconstruction COMMAND surface_test)

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET surface_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()

endif()

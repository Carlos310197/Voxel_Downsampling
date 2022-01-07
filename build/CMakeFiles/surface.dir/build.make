# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build

# Include any dependencies generated for this target.
include CMakeFiles/surface.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/surface.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/surface.dir/flags.make

CMakeFiles/surface.dir/surface_reconstruction.cpp.o: CMakeFiles/surface.dir/flags.make
CMakeFiles/surface.dir/surface_reconstruction.cpp.o: ../surface_reconstruction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/surface.dir/surface_reconstruction.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/surface.dir/surface_reconstruction.cpp.o -c /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/surface_reconstruction.cpp

CMakeFiles/surface.dir/surface_reconstruction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surface.dir/surface_reconstruction.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/surface_reconstruction.cpp > CMakeFiles/surface.dir/surface_reconstruction.cpp.i

CMakeFiles/surface.dir/surface_reconstruction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surface.dir/surface_reconstruction.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/surface_reconstruction.cpp -o CMakeFiles/surface.dir/surface_reconstruction.cpp.s

CMakeFiles/surface.dir/surface_reconstruction.cpp.o.requires:

.PHONY : CMakeFiles/surface.dir/surface_reconstruction.cpp.o.requires

CMakeFiles/surface.dir/surface_reconstruction.cpp.o.provides: CMakeFiles/surface.dir/surface_reconstruction.cpp.o.requires
	$(MAKE) -f CMakeFiles/surface.dir/build.make CMakeFiles/surface.dir/surface_reconstruction.cpp.o.provides.build
.PHONY : CMakeFiles/surface.dir/surface_reconstruction.cpp.o.provides

CMakeFiles/surface.dir/surface_reconstruction.cpp.o.provides.build: CMakeFiles/surface.dir/surface_reconstruction.cpp.o


CMakeFiles/surface.dir/voxel_structure.cu.o: CMakeFiles/surface.dir/flags.make
CMakeFiles/surface.dir/voxel_structure.cu.o: ../voxel_structure.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/surface.dir/voxel_structure.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/voxel_structure.cu -o CMakeFiles/surface.dir/voxel_structure.cu.o

CMakeFiles/surface.dir/voxel_structure.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/surface.dir/voxel_structure.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/surface.dir/voxel_structure.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/surface.dir/voxel_structure.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/surface.dir/voxel_structure.cu.o.requires:

.PHONY : CMakeFiles/surface.dir/voxel_structure.cu.o.requires

CMakeFiles/surface.dir/voxel_structure.cu.o.provides: CMakeFiles/surface.dir/voxel_structure.cu.o.requires
	$(MAKE) -f CMakeFiles/surface.dir/build.make CMakeFiles/surface.dir/voxel_structure.cu.o.provides.build
.PHONY : CMakeFiles/surface.dir/voxel_structure.cu.o.provides

CMakeFiles/surface.dir/voxel_structure.cu.o.provides.build: CMakeFiles/surface.dir/voxel_structure.cu.o


# Object files for target surface
surface_OBJECTS = \
"CMakeFiles/surface.dir/surface_reconstruction.cpp.o" \
"CMakeFiles/surface.dir/voxel_structure.cu.o"

# External object files for target surface
surface_EXTERNAL_OBJECTS =

libsurface.a: CMakeFiles/surface.dir/surface_reconstruction.cpp.o
libsurface.a: CMakeFiles/surface.dir/voxel_structure.cu.o
libsurface.a: CMakeFiles/surface.dir/build.make
libsurface.a: CMakeFiles/surface.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libsurface.a"
	$(CMAKE_COMMAND) -P CMakeFiles/surface.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/surface.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/surface.dir/build: libsurface.a

.PHONY : CMakeFiles/surface.dir/build

CMakeFiles/surface.dir/requires: CMakeFiles/surface.dir/surface_reconstruction.cpp.o.requires
CMakeFiles/surface.dir/requires: CMakeFiles/surface.dir/voxel_structure.cu.o.requires

.PHONY : CMakeFiles/surface.dir/requires

CMakeFiles/surface.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/surface.dir/cmake_clean.cmake
.PHONY : CMakeFiles/surface.dir/clean

CMakeFiles/surface.dir/depend:
	cd /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build /home/tumi/Desktop/Ouster_surf_reconst/Voxel_downsampling/build/CMakeFiles/surface.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/surface.dir/depend


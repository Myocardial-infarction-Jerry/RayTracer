# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.0/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/qiu_nangong/Documents/GitHub/RayTracer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/qiu_nangong/Documents/GitHub/RayTracer/build

# Include any dependencies generated for this target.
include CMakeFiles/ray.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ray.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ray.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ray.dir/flags.make

CMakeFiles/ray.dir/src/ray.cpp.o: CMakeFiles/ray.dir/flags.make
CMakeFiles/ray.dir/src/ray.cpp.o: /Users/qiu_nangong/Documents/GitHub/RayTracer/src/ray.cpp
CMakeFiles/ray.dir/src/ray.cpp.o: CMakeFiles/ray.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/qiu_nangong/Documents/GitHub/RayTracer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ray.dir/src/ray.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ray.dir/src/ray.cpp.o -MF CMakeFiles/ray.dir/src/ray.cpp.o.d -o CMakeFiles/ray.dir/src/ray.cpp.o -c /Users/qiu_nangong/Documents/GitHub/RayTracer/src/ray.cpp

CMakeFiles/ray.dir/src/ray.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ray.dir/src/ray.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/qiu_nangong/Documents/GitHub/RayTracer/src/ray.cpp > CMakeFiles/ray.dir/src/ray.cpp.i

CMakeFiles/ray.dir/src/ray.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ray.dir/src/ray.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/qiu_nangong/Documents/GitHub/RayTracer/src/ray.cpp -o CMakeFiles/ray.dir/src/ray.cpp.s

# Object files for target ray
ray_OBJECTS = \
"CMakeFiles/ray.dir/src/ray.cpp.o"

# External object files for target ray
ray_EXTERNAL_OBJECTS =

libray.a: CMakeFiles/ray.dir/src/ray.cpp.o
libray.a: CMakeFiles/ray.dir/build.make
libray.a: CMakeFiles/ray.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/qiu_nangong/Documents/GitHub/RayTracer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libray.a"
	$(CMAKE_COMMAND) -P CMakeFiles/ray.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ray.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ray.dir/build: libray.a
.PHONY : CMakeFiles/ray.dir/build

CMakeFiles/ray.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ray.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ray.dir/clean

CMakeFiles/ray.dir/depend:
	cd /Users/qiu_nangong/Documents/GitHub/RayTracer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/qiu_nangong/Documents/GitHub/RayTracer /Users/qiu_nangong/Documents/GitHub/RayTracer /Users/qiu_nangong/Documents/GitHub/RayTracer/build /Users/qiu_nangong/Documents/GitHub/RayTracer/build /Users/qiu_nangong/Documents/GitHub/RayTracer/build/CMakeFiles/ray.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ray.dir/depend


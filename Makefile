# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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
CMAKE_BINARY_DIR = /Users/qiu_nangong/Documents/GitHub/RayTracer

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake cache editor..."
	/opt/homebrew/Cellar/cmake/3.28.0/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	/opt/homebrew/Cellar/cmake/3.28.0/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/qiu_nangong/Documents/GitHub/RayTracer/CMakeFiles /Users/qiu_nangong/Documents/GitHub/RayTracer//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/qiu_nangong/Documents/GitHub/RayTracer/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named vec3

# Build rule for target.
vec3: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 vec3
.PHONY : vec3

# fast build rule for target.
vec3/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/vec3.dir/build.make CMakeFiles/vec3.dir/build
.PHONY : vec3/fast

#=============================================================================
# Target rules for targets named render

# Build rule for target.
render: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 render
.PHONY : render

# fast build rule for target.
render/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/render.dir/build.make CMakeFiles/render.dir/build
.PHONY : render/fast

#=============================================================================
# Target rules for targets named camera

# Build rule for target.
camera: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 camera
.PHONY : camera

# fast build rule for target.
camera/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/camera.dir/build.make CMakeFiles/camera.dir/build
.PHONY : camera/fast

#=============================================================================
# Target rules for targets named entity

# Build rule for target.
entity: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 entity
.PHONY : entity

# fast build rule for target.
entity/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/entity.dir/build.make CMakeFiles/entity.dir/build
.PHONY : entity/fast

#=============================================================================
# Target rules for targets named aabb

# Build rule for target.
aabb: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 aabb
.PHONY : aabb

# fast build rule for target.
aabb/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/aabb.dir/build.make CMakeFiles/aabb.dir/build
.PHONY : aabb/fast

#=============================================================================
# Target rules for targets named fragment

# Build rule for target.
fragment: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 fragment
.PHONY : fragment

# fast build rule for target.
fragment/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/fragment.dir/build.make CMakeFiles/fragment.dir/build
.PHONY : fragment/fast

#=============================================================================
# Target rules for targets named ray

# Build rule for target.
ray: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 ray
.PHONY : ray

# fast build rule for target.
ray/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ray.dir/build.make CMakeFiles/ray.dir/build
.PHONY : ray/fast

#=============================================================================
# Target rules for targets named scene

# Build rule for target.
scene: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 scene
.PHONY : scene

# fast build rule for target.
scene/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/scene.dir/build.make CMakeFiles/scene.dir/build
.PHONY : scene/fast

#=============================================================================
# Target rules for targets named RayTracer

# Build rule for target.
RayTracer: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 RayTracer
.PHONY : RayTracer

# fast build rule for target.
RayTracer/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RayTracer.dir/build.make CMakeFiles/RayTracer.dir/build
.PHONY : RayTracer/fast

src/aabb.o: src/aabb.cpp.o
.PHONY : src/aabb.o

# target to build an object file
src/aabb.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/aabb.dir/build.make CMakeFiles/aabb.dir/src/aabb.cpp.o
.PHONY : src/aabb.cpp.o

src/aabb.i: src/aabb.cpp.i
.PHONY : src/aabb.i

# target to preprocess a source file
src/aabb.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/aabb.dir/build.make CMakeFiles/aabb.dir/src/aabb.cpp.i
.PHONY : src/aabb.cpp.i

src/aabb.s: src/aabb.cpp.s
.PHONY : src/aabb.s

# target to generate assembly for a file
src/aabb.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/aabb.dir/build.make CMakeFiles/aabb.dir/src/aabb.cpp.s
.PHONY : src/aabb.cpp.s

src/camera.o: src/camera.cpp.o
.PHONY : src/camera.o

# target to build an object file
src/camera.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/camera.dir/build.make CMakeFiles/camera.dir/src/camera.cpp.o
.PHONY : src/camera.cpp.o

src/camera.i: src/camera.cpp.i
.PHONY : src/camera.i

# target to preprocess a source file
src/camera.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/camera.dir/build.make CMakeFiles/camera.dir/src/camera.cpp.i
.PHONY : src/camera.cpp.i

src/camera.s: src/camera.cpp.s
.PHONY : src/camera.s

# target to generate assembly for a file
src/camera.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/camera.dir/build.make CMakeFiles/camera.dir/src/camera.cpp.s
.PHONY : src/camera.cpp.s

src/entity.o: src/entity.cpp.o
.PHONY : src/entity.o

# target to build an object file
src/entity.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/entity.dir/build.make CMakeFiles/entity.dir/src/entity.cpp.o
.PHONY : src/entity.cpp.o

src/entity.i: src/entity.cpp.i
.PHONY : src/entity.i

# target to preprocess a source file
src/entity.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/entity.dir/build.make CMakeFiles/entity.dir/src/entity.cpp.i
.PHONY : src/entity.cpp.i

src/entity.s: src/entity.cpp.s
.PHONY : src/entity.s

# target to generate assembly for a file
src/entity.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/entity.dir/build.make CMakeFiles/entity.dir/src/entity.cpp.s
.PHONY : src/entity.cpp.s

src/fragment.o: src/fragment.cpp.o
.PHONY : src/fragment.o

# target to build an object file
src/fragment.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/fragment.dir/build.make CMakeFiles/fragment.dir/src/fragment.cpp.o
.PHONY : src/fragment.cpp.o

src/fragment.i: src/fragment.cpp.i
.PHONY : src/fragment.i

# target to preprocess a source file
src/fragment.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/fragment.dir/build.make CMakeFiles/fragment.dir/src/fragment.cpp.i
.PHONY : src/fragment.cpp.i

src/fragment.s: src/fragment.cpp.s
.PHONY : src/fragment.s

# target to generate assembly for a file
src/fragment.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/fragment.dir/build.make CMakeFiles/fragment.dir/src/fragment.cpp.s
.PHONY : src/fragment.cpp.s

src/main.o: src/main.cpp.o
.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RayTracer.dir/build.make CMakeFiles/RayTracer.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RayTracer.dir/build.make CMakeFiles/RayTracer.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/RayTracer.dir/build.make CMakeFiles/RayTracer.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/ray.o: src/ray.cpp.o
.PHONY : src/ray.o

# target to build an object file
src/ray.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ray.dir/build.make CMakeFiles/ray.dir/src/ray.cpp.o
.PHONY : src/ray.cpp.o

src/ray.i: src/ray.cpp.i
.PHONY : src/ray.i

# target to preprocess a source file
src/ray.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ray.dir/build.make CMakeFiles/ray.dir/src/ray.cpp.i
.PHONY : src/ray.cpp.i

src/ray.s: src/ray.cpp.s
.PHONY : src/ray.s

# target to generate assembly for a file
src/ray.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ray.dir/build.make CMakeFiles/ray.dir/src/ray.cpp.s
.PHONY : src/ray.cpp.s

src/render.o: src/render.cpp.o
.PHONY : src/render.o

# target to build an object file
src/render.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/render.dir/build.make CMakeFiles/render.dir/src/render.cpp.o
.PHONY : src/render.cpp.o

src/render.i: src/render.cpp.i
.PHONY : src/render.i

# target to preprocess a source file
src/render.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/render.dir/build.make CMakeFiles/render.dir/src/render.cpp.i
.PHONY : src/render.cpp.i

src/render.s: src/render.cpp.s
.PHONY : src/render.s

# target to generate assembly for a file
src/render.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/render.dir/build.make CMakeFiles/render.dir/src/render.cpp.s
.PHONY : src/render.cpp.s

src/scene.o: src/scene.cpp.o
.PHONY : src/scene.o

# target to build an object file
src/scene.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/scene.dir/build.make CMakeFiles/scene.dir/src/scene.cpp.o
.PHONY : src/scene.cpp.o

src/scene.i: src/scene.cpp.i
.PHONY : src/scene.i

# target to preprocess a source file
src/scene.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/scene.dir/build.make CMakeFiles/scene.dir/src/scene.cpp.i
.PHONY : src/scene.cpp.i

src/scene.s: src/scene.cpp.s
.PHONY : src/scene.s

# target to generate assembly for a file
src/scene.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/scene.dir/build.make CMakeFiles/scene.dir/src/scene.cpp.s
.PHONY : src/scene.cpp.s

src/vec3.o: src/vec3.cpp.o
.PHONY : src/vec3.o

# target to build an object file
src/vec3.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/vec3.dir/build.make CMakeFiles/vec3.dir/src/vec3.cpp.o
.PHONY : src/vec3.cpp.o

src/vec3.i: src/vec3.cpp.i
.PHONY : src/vec3.i

# target to preprocess a source file
src/vec3.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/vec3.dir/build.make CMakeFiles/vec3.dir/src/vec3.cpp.i
.PHONY : src/vec3.cpp.i

src/vec3.s: src/vec3.cpp.s
.PHONY : src/vec3.s

# target to generate assembly for a file
src/vec3.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/vec3.dir/build.make CMakeFiles/vec3.dir/src/vec3.cpp.s
.PHONY : src/vec3.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... RayTracer"
	@echo "... aabb"
	@echo "... camera"
	@echo "... entity"
	@echo "... fragment"
	@echo "... ray"
	@echo "... render"
	@echo "... scene"
	@echo "... vec3"
	@echo "... src/aabb.o"
	@echo "... src/aabb.i"
	@echo "... src/aabb.s"
	@echo "... src/camera.o"
	@echo "... src/camera.i"
	@echo "... src/camera.s"
	@echo "... src/entity.o"
	@echo "... src/entity.i"
	@echo "... src/entity.s"
	@echo "... src/fragment.o"
	@echo "... src/fragment.i"
	@echo "... src/fragment.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/ray.o"
	@echo "... src/ray.i"
	@echo "... src/ray.s"
	@echo "... src/render.o"
	@echo "... src/render.i"
	@echo "... src/render.s"
	@echo "... src/scene.o"
	@echo "... src/scene.i"
	@echo "... src/scene.s"
	@echo "... src/vec3.o"
	@echo "... src/vec3.i"
	@echo "... src/vec3.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system


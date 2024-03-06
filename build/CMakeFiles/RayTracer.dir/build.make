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
include CMakeFiles/RayTracer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/RayTracer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/RayTracer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RayTracer.dir/flags.make

CMakeFiles/RayTracer.dir/src/main.cpp.o: CMakeFiles/RayTracer.dir/flags.make
CMakeFiles/RayTracer.dir/src/main.cpp.o: /Users/qiu_nangong/Documents/GitHub/RayTracer/src/main.cpp
CMakeFiles/RayTracer.dir/src/main.cpp.o: CMakeFiles/RayTracer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/qiu_nangong/Documents/GitHub/RayTracer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RayTracer.dir/src/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RayTracer.dir/src/main.cpp.o -MF CMakeFiles/RayTracer.dir/src/main.cpp.o.d -o CMakeFiles/RayTracer.dir/src/main.cpp.o -c /Users/qiu_nangong/Documents/GitHub/RayTracer/src/main.cpp

CMakeFiles/RayTracer.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RayTracer.dir/src/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/qiu_nangong/Documents/GitHub/RayTracer/src/main.cpp > CMakeFiles/RayTracer.dir/src/main.cpp.i

CMakeFiles/RayTracer.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RayTracer.dir/src/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/qiu_nangong/Documents/GitHub/RayTracer/src/main.cpp -o CMakeFiles/RayTracer.dir/src/main.cpp.s

# Object files for target RayTracer
RayTracer_OBJECTS = \
"CMakeFiles/RayTracer.dir/src/main.cpp.o"

# External object files for target RayTracer
RayTracer_EXTERNAL_OBJECTS =

RayTracer: CMakeFiles/RayTracer.dir/src/main.cpp.o
RayTracer: CMakeFiles/RayTracer.dir/build.make
RayTracer: libvec3.a
RayTracer: librender.a
RayTracer: libcamera.a
RayTracer: libentity.a
RayTracer: libfragment.a
RayTracer: libray.a
RayTracer: libscene.a
RayTracer: /opt/homebrew/lib/libopencv_gapi.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_stitching.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_alphamat.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_aruco.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_bgsegm.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_bioinspired.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_ccalib.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_dnn_objdetect.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_dnn_superres.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_dpm.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_face.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_freetype.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_fuzzy.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_hfs.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_img_hash.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_intensity_transform.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_line_descriptor.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_mcc.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_quality.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_rapid.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_reg.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_rgbd.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_saliency.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_sfm.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_stereo.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_structured_light.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_superres.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_surface_matching.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_tracking.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_videostab.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_viz.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_wechat_qrcode.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_xfeatures2d.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_xobjdetect.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_xphoto.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_shape.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_highgui.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_datasets.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_plot.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_text.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_ml.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_phase_unwrapping.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_optflow.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_ximgproc.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_video.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_videoio.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_imgcodecs.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_objdetect.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_calib3d.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_dnn.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_features2d.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_flann.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_photo.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_imgproc.4.9.0.dylib
RayTracer: /opt/homebrew/lib/libopencv_core.4.9.0.dylib
RayTracer: CMakeFiles/RayTracer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/qiu_nangong/Documents/GitHub/RayTracer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable RayTracer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RayTracer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RayTracer.dir/build: RayTracer
.PHONY : CMakeFiles/RayTracer.dir/build

CMakeFiles/RayTracer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RayTracer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RayTracer.dir/clean

CMakeFiles/RayTracer.dir/depend:
	cd /Users/qiu_nangong/Documents/GitHub/RayTracer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/qiu_nangong/Documents/GitHub/RayTracer /Users/qiu_nangong/Documents/GitHub/RayTracer /Users/qiu_nangong/Documents/GitHub/RayTracer/build /Users/qiu_nangong/Documents/GitHub/RayTracer/build /Users/qiu_nangong/Documents/GitHub/RayTracer/build/CMakeFiles/RayTracer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/RayTracer.dir/depend

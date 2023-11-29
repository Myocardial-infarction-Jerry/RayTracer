# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++11 -Iinclude
CFLAGS += -I/usr/local/cuda/include

# CUDA Compiler
NVCC = nvcc

# CUDA Compiler flags
NVCCFLAGS = -std=c++11 -Iinclude

# Debug flags
DEBUGFLAGS = -g -DDEBUG

# Source files
SRCS = $(wildcard src/*.cpp)
CU_SRCS = $(wildcard src/*.cu)

# Object files
OBJS = $(SRCS:.cpp=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)

# Executable
EXEC = myproject

# Default target
all: $(EXEC)

# Compile C++ source files
%.o: %.cpp
	$(CC) $(CFLAGS) $(DEBUGFLAGS) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(DEBUGFLAGS) -c $< -o $@

# Link object files into executable
$(EXEC): $(OBJS) $(CU_OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(CU_OBJS) -o $@

# Debug target
debug: $(EXEC)
	gdb $(EXEC)

# Clean
clean:
	rm -f $(OBJS) $(CU_OBJS) $(EXEC)
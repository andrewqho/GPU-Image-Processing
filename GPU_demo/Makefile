# CS 179 Final Project Makefile
# Written by Andrew Ho, 2021
#

# Product Names
CUDA_OBJ = cuda.o

# Input Names
CUDA_FILES = src/polyline_gpu.cu
CPP_FILES = src/polyline_cpu.cpp src/main.cpp

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda-9.1
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS = 
NVCC_GENCODES = -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# CUDA Object Files
CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lsndfile $(OPENCV)

# CPP Object Files
CPP_OBJ_FILES = $(notdir $(addsuffix .o, $(CPP_FILES)))

# ------------------------------------------------------------------------------
# Make Rules
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ_CPP = $(notdir $(addsuffix .o, $(CPP_FILES)))

# Top level rules
all: main

test: test.cpp.o polyline_cpu.cpp.o polyline_gpu.cu.o cuda.o
	$(GPP) $(FLAGS) -O MAIN $(INCLUDE) $^ $(LIBS)

main: $(OBJ_CPP) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o main $(INCLUDE) $^ $(LIBS)

# Compile C++ Source Files
test.cpp.o: src/main.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 

main.cpp.o: src/main.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 

polyline_cpu.cpp.o: src/polyline_cpu.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 


# Compile CUDA Source Files
polyline_gpu.cu.o: src/polyline_gpu.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f main main.cpp.o polyline_cpu.cpp.o polyline_gpu.cu.o tester.cpp.o cuda.o
	rm -r images/output/*

.PHONY: clean

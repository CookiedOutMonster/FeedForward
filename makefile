# Compiler settings
NVCC = nvcc
CXX = g++

# Directories
SRC_DIR = neural_net
TEST_DIR = tests
LIB_DIR = lib
IMAGE_READER_DIR = image_reader

# Google Test settings
GTEST_INCLUDE = $(LIB_DIR)/include
GTEST_LIB = $(LIB_DIR)/libgtest.a
GTEST_MAIN_LIB = $(LIB_DIR)/libgtest_main.a

# Source files - include both .cu and .cpp files
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(IMAGE_READER_DIR)/*.cpp)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cu) $(wildcard $(TEST_DIR)/*.cpp)

# Object files
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
TEST_OBJS := $(TEST_OBJS:.cu=.o)

# Get test names without extension for individual test targets
TEST_NAMES = $(basename $(notdir $(TEST_SRCS)))

# Compiler flags
CUDA_ARCH = -arch=sm_60  # Adjust based on your GPU architecture
NVCCFLAGS = $(CUDA_ARCH) --std=c++17 -O0 -g -I$(GTEST_INCLUDE) -I$(SRC_DIR) -I$(IMAGE_READER_DIR) -MMD -MP -Xcompiler "-fopenmp"
CXXFLAGS = -std=c++17 -O0 -g -pthread -I$(GTEST_INCLUDE) -I$(SRC_DIR) -I$(IMAGE_READER_DIR) -MMD -MP -fopenmp

# Linker flags
CXX_LDFLAGS = -L$(LIB_DIR) -lgtest -lgtest_main -pthread -fopenmp
NVCC_LDFLAGS = -L$(LIB_DIR) -lgtest -lgtest_main -lcudart -Xcompiler "-fopenmp"

# Default target
all: $(TEST_NAMES)

# Main program
main: Main.cu $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(NVCC_LDFLAGS)

# Build test executables
$(TEST_NAMES): %: $(TEST_DIR)/%.o $(CUDA_OBJS) $(CPP_OBJS)
	$(NVCC) $^ -o $@ $(NVCC_LDFLAGS)

# Build test objects
$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST_DIR)/%.o: $(TEST_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Build CUDA objects
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Build C++ objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run individual tests
run_%: %
	./$< --gtest_catch_exceptions=0

# Run all tests
run_tests: $(TEST_NAMES)
	for test in $(TEST_NAMES); do ./$$test --gtest_catch_exceptions=0; done

# Run main program
run_main: main
	./main

# Debug mode
debug: CXXFLAGS += -DDEBUG
debug: NVCCFLAGS += -DDEBUG
debug: all

# Debug specific target
debug_%: %
	gdb ./$<

# Clean up
clean:
	rm -f $(CUDA_OBJS) $(CPP_OBJS) $(TEST_OBJS) $(TEST_NAMES) main

# Declare phony targets
.PHONY: all clean run_tests run_main $(addprefix run_,$(TEST_NAMES))

# Include dependency files
-include $(CUDA_OBJS:.o=.d) $(CPP_OBJS:.o=.d) $(TEST_OBJS:.o=.d)
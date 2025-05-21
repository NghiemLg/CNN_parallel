cmake_minimum_required(VERSION 3.10)
project(CNN_CUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust this based on your GPU architecture

# Enable CUDA error checking
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add project include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    kernel.cu
)

# Create executable
add_executable(cnn_cuda ${SOURCES})
set_target_properties(cnn_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(cnn_cuda 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
)

# Add CUDA error checking macro
target_compile_definitions(cnn_cuda PRIVATE
    CUDA_ERROR_CHECK
)

# Set CUDA include directories
target_include_directories(cnn_cuda PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
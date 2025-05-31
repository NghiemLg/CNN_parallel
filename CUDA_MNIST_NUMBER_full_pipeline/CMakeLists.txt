cmake_minimum_required(VERSION 3.10)
project(ParallelCNN_CUDA LANGUAGES CXX CUDA)

# Tìm thư viện CUDA
find_package(CUDA REQUIRED)
# Tìm thư viện OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 86) # Kiến trúc Ampere cho RTX 3050

# Thêm cờ tối ưu hóa
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# Tạo executable cho training
add_executable(cnn_cuda main.cu layer.cu)
target_include_directories(cnn_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cnn_cuda PRIVATE ${CUDA_LIBRARIES} cudart cublas cuda)

# Tạo executable cho test thực tế
add_executable(test_real test_real.cu layer.cu)
target_include_directories(test_real PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_real PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS} cudart cublas cuda)
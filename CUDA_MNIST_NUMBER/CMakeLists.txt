cmake_minimum_required(VERSION 3.10)
project(ParallelCNN_CUDA LANGUAGES CXX CUDA)

# Tìm thư viện CUDA
find_package(CUDA REQUIRED)
# Tìm thư viện OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Thêm các file nguồn
set(SOURCES
    main.cu
    layer.cu
    mnist.h
    layer_c.h
)

# Tạo executable cho training
add_executable(cnn_cuda main.cu layer.cu)
target_link_libraries(cnn_cuda PRIVATE ${CUDA_LIBRARIES})

# Tạo executable cho test thực tế
add_executable(test_real test_real.cu layer.cu)
target_link_libraries(test_real PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Bao gồm thư mục hiện tại để tìm header
target_include_directories(cnn_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_real PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

# Liên kết thêm các thư viện CUDA cần thiết
target_link_libraries(cnn_cuda PRIVATE cudart cublas cuda)
target_link_libraries(test_real PRIVATE cudart cublas cuda)
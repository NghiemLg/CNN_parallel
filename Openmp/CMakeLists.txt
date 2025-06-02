cmake_minimum_required(VERSION 3.10)
project(ParallelCNN_OpenMP LANGUAGES CXX)

# Tìm OpenMP
find_package(OpenMP REQUIRED)
# Tìm OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

# Tạo executable cho training (chỉ Main.cpp)
add_executable(cnn_omp Main.cpp)
target_link_libraries(cnn_omp PRIVATE OpenMP::OpenMP_CXX)


# Bao gồm thư mục hiện tại để tìm header
target_include_directories(cnn_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cnn_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS}) 
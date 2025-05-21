# CNN_CUDA - Paraellel parallel programming

Dự án này tập trung vào việc so sánh hiệu năng của các mạng nơ-ron tích chập (CNN) khi chạy trên CPU và GPU, sử dụng các công nghệ song song hóa khác nhau.

## Cấu trúc thư mục

### 1. only_Convolution_CPU-GPU_compare/
Thư mục này chứa mã nguồn và các thử nghiệm so sánh hiệu năng của phép tích chập (convolution) khi chạy trên CPU và GPU. Các file chính bao gồm:
- `kernel.cu`: Chứa các kernel CUDA cho phép tích chập
- `ConvolutionalLayer.h`: Định nghĩa lớp tích chập
- `PoolingLayer.h`: Định nghĩa lớp pooling
- `Helpers.h`: Các hàm tiện ích và helper
- `Filters.h`: Định nghĩa các bộ lọc cho CNN
- `guide.MD`: Hướng dẫn sử dụng và biên dịch

### 2. Openmp/
Thư mục chứa phiên bản song song hóa của CNN sử dụng OpenMP. Các file chính:
- `Main.cpp`: File chính chứa hàm main và logic chương trình
- `layer.h`: Định nghĩa các lớp CNN và các hàm xử lý
- `mnist.h`: Định nghĩa cấu trúc dữ liệu và hàm xử lý cho tập MNIST
- `CMakeLists.txt`: File cấu hình CMake cho việc biên dịch

### 3. CUDA_MNIST_NUMBER_full_pipeline/
Thư mục chứa pipeline hoàn chỉnh để huấn luyện và đánh giá mô hình CNN trên MNIST sử dụng CUDA:
- `main.cu`: File chính của chương trình
- `layer.cu` và `layer_c.h`: Định nghĩa các lớp CNN và các hàm CUDA
- `test_real.cu`: File test với dữ liệu thực tế
- Các file trọng số đã huấn luyện: `weights_c1.bin`, `weights_s1.bin`, `weights_f.bin`
- Các file bias: `bias_c1.bin`, `bias_s1.bin`, `bias_f.bin`
- Các file ảnh test: `image.png`, `image copy *.png`

### 4. data/
Thư mục chứa tập dữ liệu MNIST:
- `train-images.idx3-ubyte`: Ảnh huấn luyện
- `train-labels.idx1-ubyte`: Nhãn huấn luyện
- `t10k-images.idx3-ubyte`: Ảnh kiểm thử
- `t10k-labels.idx1-ubyte`: Nhãn kiểm thử

### 5. Parallelizing_Convolutional_Neural_Networks.pdf
Tài liệu nghiên cứu mô tả chi tiết về các phương pháp song song hóa CNN và kết quả thử nghiệm.

## Yêu cầu hệ thống
- CUDA Toolkit (phiên bản mới nhất)
- OpenMP
- CMake (phiên bản 3.0 trở lên)
- Compiler hỗ trợ C++11
- GPU hỗ trợ CUDA

## Cách biên dịch và chạy
Mỗi thư mục con đều có file CMakeLists.txt riêng. Để biên dịch:
```bash
mkdir build
cd build
cmake ..
make
```

## Mục đích
Dự án này nhằm:
1. So sánh hiệu năng của phép tích chập trên CPU (sử dụng OpenMP) và GPU (sử dụng CUDA)
2. Đánh giá các phương pháp song song hóa khác nhau
3. Cung cấp một pipeline hoàn chỉnh để huấn luyện và đánh giá CNN trên MNIST
4. Tối ưu hóa hiệu suất của các mô hình CNN thông qua việc song song hóa 
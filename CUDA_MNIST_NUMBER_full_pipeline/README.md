# Triển Khai CNN Song Song Sử Dụng CUDA

Dự án này triển khai một Convolutional Neural Network (CNN) để nhận dạng chữ số MNIST sử dụng NVIDIA CUDA cho xử lý song song. Việc triển khai này cho thấy cách GPU acceleration có thể cải thiện đáng kể hiệu suất training và inference của các deep learning models.

## Cấu Trúc Dự Án

```
CUDA_MNIST_NUMBER_full_pipeline/
├── main.cu           # File chương trình CUDA chính
├── layer.cu          # Triển khai CUDA kernels
├── layer_c.h         # Khai báo Layer class và kernels
├── mnist.h           # MNIST dataset loader
├── test_real.cu      # Chương trình test với ảnh thực tế
├── CMakeLists.txt    # File cấu hình CMake
├── information.MD    # Thông tin chi tiết về dự án
├── README.md         # File này
├── weights_c1.bin    # Weights của convolutional layer
├── weights_s1.bin    # Weights của subsampling layer
├── weights_f.bin     # Weights của fully connected layer
├── bias_c1.bin       # Bias của convolutional layer
├── bias_s1.bin       # Bias của subsampling layer
├── bias_f.bin        # Bias của fully connected layer
├── image.png         # Ảnh mẫu để test
├── image copy 0.png  # Ảnh test số 0
├── image copy 1.png  # Ảnh test số 1
├── image copy 4.png  # Ảnh test số 4
└── image copy 5.png  # Ảnh test số 5
```

## Yêu Cầu Hệ Thống

- GPU NVIDIA hỗ trợ CUDA
- CUDA Toolkit (phiên bản 12.4 trở lên)
- Trình biên dịch tương thích C++11
- CMake (phiên bản 3.10 trở lên)

## Kiến Trúc Mạng

CNN bao gồm các layers sau:
1. Input Layer (28x28)
2. Convolutional Layer (C1)
   - 6 feature maps
   - 5x5 kernel size
   - Output: 24x24x6
3. Subsampling Layer (S1)
   - 4x4 kernel size
   - Output: 6x6x6
4. Fully Connected Layer (F)
   - 10 output neurons (digits 0-9)

## Chi Tiết Triển Khai CUDA

### Các Thành Phần Chính

1. **Layer Class**
   - Quản lý weights, biases và activations
   - Xử lý memory allocation cho GPU
   - Hỗ trợ forward và backward propagation

2. **CUDA Kernels**
   - Forward Propagation:
     - `fp_c1`: Convolutional layer
     - `fp_s1`: Subsampling layer
     - `fp_f`: Fully connected layer
   - Backward Propagation:
     - `bp_f`: Fully connected layer gradients
     - `bp_s1`: Subsampling layer gradients
     - `bp_c1`: Convolutional layer gradients

3. **Memory Management**
   - Sử dụng CUDA device memory cho weights và activations
   - Triển khai efficient memory transfers giữa CPU và GPU

## Biên Dịch

```bash
mkdir build
cd build
cmake ..
make
```

## Sử Dụng

1. Biên dịch mã nguồn sử dụng CMake như hướng dẫn trên
2. Chạy chương trình:
   ```bash
   ./cnn_cuda
   ```
3. Để nhận dạng ảnh thực tế:
   ```bash
   ./test_real image.png
   ```

## Tính Năng

- Training CNN model trên MNIST dataset
- Nhận dạng chữ số từ ảnh đầu vào
- Tối ưu hóa hiệu suất bằng CUDA
- Hỗ trợ real-time inference

## Tối Ưu Hóa

1. **Kernel Configuration**
   - Optimized block và grid sizes cho từng layer
   - Efficient memory access patterns
   - Shared memory usage khi có lợi

2. **Memory Management**
   - Giảm thiểu host-device transfers
   - Efficient memory allocation và deallocation
   - Proper error handling cho CUDA operations

## Xử Lý Sự Cố

1. **CUDA Header Issues**
   - Đảm bảo CUDA toolkit được cài đặt đúng cách
   - Kiểm tra include paths trong lệnh biên dịch
   - Verify CUDA version compatibility

2. **Memory Issues**
   - Kiểm tra GPU memory availability
   - Monitor memory usage trong quá trình thực thi
   - Handle CUDA errors appropriately

## Đóng Góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## Giấy Phép

Dự án này được cấp phép theo các điều khoản trong file LICENSE trong thư mục gốc. 
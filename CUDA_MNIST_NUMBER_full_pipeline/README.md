# Parallel CNN Implementation using CUDA

This project implements a Convolutional Neural Network (CNN) for MNIST digit recognition using NVIDIA CUDA for parallel processing. The implementation demonstrates how GPU acceleration can significantly improve the training and inference performance of deep learning models.

## Project Structure

```
CUDA/
├── main.cu           # Main CUDA program file
├── layer.cu          # CUDA kernel implementations
├── layer_c.h         # Layer class and kernel declarations
├── mnist.h           # MNIST dataset loader
└── README.md         # This file
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 12.4 or later)
- C++11 compatible compiler
- MNIST dataset files in the `../data/` directory:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## Network Architecture

The CNN consists of the following layers:
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

## CUDA Implementation Details

### Key Components

1. **Layer Class**
   - Manages weights, biases, and activations
   - Handles memory allocation for GPU
   - Supports forward and backward propagation

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
   - Uses CUDA device memory for weights and activations
   - Implements efficient memory transfers between CPU and GPU

## Compilation

```bash
nvcc -o cnn_cuda main.cu layer.cu -std=c++11 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda
```

## Usage

1. Ensure MNIST dataset files are in the correct location
2. Compile the code using the command above
3. Run the executable:
   ```bash
   ./cnn_cuda
   ```

## Performance

The CUDA implementation achieves:
- Training time: ~2.5 seconds
- Error rate: ~7.8%
- Significant speedup compared to CPU implementations

## Optimization Techniques

1. **Kernel Configuration**
   - Optimized block and grid sizes for each layer
   - Efficient memory access patterns
   - Shared memory usage where beneficial

2. **Memory Management**
   - Minimized host-device transfers
   - Efficient memory allocation and deallocation
   - Proper error handling for CUDA operations

## Troubleshooting

1. **CUDA Header Issues**
   - Ensure CUDA toolkit is properly installed
   - Check include paths in compilation command
   - Verify CUDA version compatibility

2. **Memory Issues**
   - Check GPU memory availability
   - Monitor memory usage during execution
   - Handle CUDA errors appropriately

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the terms of the LICENSE file in the root directory. 
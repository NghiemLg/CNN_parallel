I'll provide a comprehensive guide on how to run the project and understand its results.

### 1. Project Overview
This is a CUDA-based CNN (Convolutional Neural Network) implementation that demonstrates:
- CPU vs GPU performance comparison for CNN operations
- Implementation of Convolutional and Pooling layers
- Image processing using OpenCV

### 2. Prerequisites
The project requires:
- CUDA Toolkit 12.0 or compatible
- OpenCV 4.7.0
- CMake 3.10 or higher
- C++17 compatible compiler
- NVIDIA GPU with at least 4GB VRAM
- System RAM: 16GB or more

### 3. Project Structure
```
CNN_CUDA/
├── data/
│   └── Animal Images/
│       ├── cats/         # Input images
│       └── dogs/         # Input images
├── kernel.cu            # Main CUDA implementation
├── ConvolutionalLayer.h # Convolution layer implementation
├── PoolingLayer.h       # Pooling layer implementation
├── Filters.h           # CNN filters
├── Helpers.h           # Utility functions
└── CMakeLists.txt      # Build configuration
```

### 4. Building the Project
1. Create and enter the build directory:
```bash
mkdir build
cd build
```

2. Configure and build:
```bash
cmake ..
make
```

3. Create a symbolic link to the data directory:
```bash
ln -s ../data .
```

### 5. Running the Project
```bash
./cnn_cuda
```

### 6. Understanding the Results

The program output has several sections:

1. **Device Information**
```
==================================================
=                  DEVICE INFO                   =
==================================================
Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability: 8.6
...
```
This shows your GPU capabilities and specifications.

2. **Image Loading**
```
==================================================
=                   LOAD IMAGE                   =
==================================================
Seccussfully loaded 1 images, could not load 0 images.
```
Shows how many images were successfully loaded from the `data/Animal Images/cats/` directory.

3. **CPU Results**
```
==================================================
=                   CPU RESULT                   =
==================================================
[CPU] Convolutional Layer took 325 ms to run.
[CPU] Pooling Layer took 49 ms to run.
```
Shows the time taken by CPU implementation:
- Convolutional Layer: 325ms
- Pooling Layer: 49ms

4. **GPU Results**
```
==================================================
=                   GPU RESULT                   =
==================================================
[GPU] Convolutional Layer - Memory Copy: 26.05 ms, Kernel: 1489.90 ms
[GPU] Pooling Layer - Memory Copy: 10.82 ms, Kernel: 255.22 ms
```
Shows the time taken by GPU implementation:
- Convolutional Layer:
  - Memory Copy: 26.05ms (CPU to GPU transfer)
  - Kernel Execution: 1489.90ms (actual computation)
- Pooling Layer:
  - Memory Copy: 10.82ms
  - Kernel Execution: 255.22ms

5. **Result Verification**
```
Check if GPU result equal to CPU result: 1
```
The program verifies that GPU results match CPU results (1 means they match).

### 7. Performance Analysis

In this run:
1. **Total CPU Time**: 374ms (325ms + 49ms)
2. **Total GPU Time**: 1781.99ms (26.05ms + 1489.90ms + 10.82ms + 255.22ms)

The GPU implementation is slower in this case because:
1. Small dataset (only 1 image)
2. Memory transfer overhead
3. Kernel launch overhead
4. The specific operations might not be large enough to benefit from GPU parallelism

### 8. Improving Performance

To see better GPU performance:
1. Process more images (uncomment line 47 in `Helpers.h` to load more images)
2. Use larger images
3. Process images in batches
4. Optimize memory transfers

### 9. Demo Mode
The program runs in demo mode by default (`DEMO_MODE true`). You can:
- Set `DEMO_MODE_SHOW_RES_IMAGE true` to visualize the processed images
- The program validates GPU results against CPU results for correctness

### 10. Memory Management
The program uses smart pointers for memory management:
```cpp
using ImageArray = std::unique_ptr<int*[]>;
using ImageArray2D = std::unique_ptr<int**>;
using ImageArray3D = std::unique_ptr<int***>;
```

This helps prevent memory leaks and ensures proper cleanup.

Would you like me to explain any specific part in more detail or help you modify the code to improve performance?

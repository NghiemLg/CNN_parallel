/**
 * @file layer.cu
 * @brief Implementation of the Layer class for CNN using CUDA
 * This file contains the implementation of neural network layers with GPU acceleration
 */

#include "layer_c.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/**
 * @class Layer
 * @brief Represents a neural network layer with GPU-accelerated operations
 * 
 * The Layer class manages memory and computations for a single layer in the CNN.
 * It handles both forward and backward propagation on the GPU.
 */

/**
 * @brief Constructor for the Layer class
 * @param M Number of neurons in the current layer
 * @param N Number of neurons in the previous layer
 * @param O Size of the output
 * 
 * Initializes the layer by allocating GPU memory for weights, biases,
 * activations, and gradients.
 */
Layer::Layer(int M, int N, int O) {
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

/**
 * @brief Destructor for the Layer class
 * 
 * Frees all allocated GPU memory to prevent memory leaks
 */
Layer::~Layer() {
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

/**
 * @brief Clears the layer's output and pre-activation values
 * 
 * Resets the output and pre-activation arrays to zero on the GPU
 */
void Layer::clear() {
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

/**
 * @brief Clears all gradients for backward propagation
 * 
 * Resets all gradient arrays to zero on the GPU
 */
void Layer::bp_clear() {
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

/**
 * @brief Activation function (sigmoid)
 * @param v Input value
 * @return Value after applying sigmoid
 */
__device__ float step_function(float v)
{
    return 1 / (1 + expf(-v));  // Using expf() for faster computation
}

/**
 * @brief CUDA kernel for forward propagation in convolutional layer
 * @param input Input feature maps
 * @param preact Pre-activation values
 * @param weight Convolutional kernels
 * @param bias Bias values
 * 
 * Performs convolution operation on the GPU
 */
__global__ void fp_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5], float bias[6]) {
    int m = blockIdx.x; // One block per output feature map
    int x = threadIdx.x; // Thread along x dimension of output feature map
    int y = threadIdx.y; // Thread along y dimension of output feature map

    if (m < 6 && x < 24 && y < 24) {
        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                sum += input[x + i][y + j] * weight[m][i][j];
            }
        }
        preact[m][x][y] = sum + bias[m];
    }
}

/**
 * @brief CUDA kernel for forward propagation in subsampling layer
 * @param input Input feature maps
 * @param preact Pre-activation values
 * @param weight Subsampling weights
 * @param bias Bias values
 * 
 * Performs subsampling operation on the GPU
 */
__global__ void fp_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float bias[1]) {
    int m = blockIdx.z;  // Use z-dimension in grid to handle different feature maps
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global x index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Calculate global y index

    if (m < 6 && x < 6 && y < 6) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {  // kernel width
            for (int j = 0; j < 4; ++j) {  // kernel height
               
                sum += weight[0][i][j] * input[m][x * 4 + i][y * 4 + j];
            }
        }
        // Add bias and store the result in the corresponding location in preact
        preact[m][x][y] = sum + bias[0];
    }
}

/**
 * @brief CUDA kernel for forward propagation in fully connected layer
 * @param input Input feature maps
 * @param preact Pre-activation values
 * @param weight Fully connected weights
 * @param bias Bias values
 * 
 * Performs fully connected layer operation on the GPU
 */
__global__ void fp_f(float input[6][6][6], float preact[10], float weight[10][6][6][6], float bias[10]) {
    int o = blockIdx.x * blockDim.x + threadIdx.x; // Index for the output dimension
    if (o < 10) {
        float sum = 0.0f;
        for (int j = 0; j < 6; ++j) { // First dimension of input
            for (int k = 0; k < 6; ++k) { // Second dimension of input
                for (int l = 0; l < 6; ++l) { // Third dimension of input
                    sum += weight[o][j][k][l] * input[j][k][l];
                }
            }
        }
        atomicAdd(&preact[o], sum); // Atomically add the sum to the output to avoid write conflicts
        preact[o] += bias[o]; // Add bias to each element
    }
}

/**
 * @brief CUDA kernel for backward propagation in fully connected layer
 * @param d_weight Weight gradients
 * @param d_bias Bias gradients
 * @param d_preact Pre-activation gradients
 * @param preact Pre-activation values
 * 
 * Computes gradients for the fully connected layer
 */
__global__ void bp_f(float d_weight[10][6][6][6], float d_bias[10], float d_preact[10], float preact[6][6][6]) {
    // Use a single shared memory buffer for the entire output matrix.
    __shared__ float shared_p_output[6][6][6];
	float dt = 1.0E-01f;
    // Load p_output into shared memory once per block
    int idx = threadIdx.x + blockDim.x * threadIdx.y;
    int total_threads = blockDim.x * blockDim.y;
    for (int index = idx; index < 6*6*6; index += total_threads) {
        int l = index % 6;
        int k = (index / 6) % 6;
        int j = index / 36;
        shared_p_output[j][k][l] = preact[j][k][l];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 10) {
        float d_preact_val = d_preact[i];
        float* d_weight_i = d_weight[i][0][0];

        // Update weights using shared output memory
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                for (int l = 0; l < 6; ++l) {
                    d_weight_i[j*36 + k*6 + l] += d_preact_val * shared_p_output[j][k][l];
                }
            }
        }
        // Update bias for this filter
        atomicAdd(&d_bias[i], dt * d_preact_val);
    }
}

/**
 * @brief CUDA kernel for backward propagation in subsampling layer
 * @param d_output Output gradients
 * @param d_preact Pre-activation gradients
 * @param weight Subsampling weights
 * @param d_weight Weight gradients
 * 
 * Computes gradients for the subsampling layer
 */
__global__ void bp_s1(float d_output[6][6][6], float d_preact[6][6][6], float weight[1][4][4], float d_weight[1][4][4]) {
    // ... existing kernel code ...
}

/**
 * @brief CUDA kernel for backward propagation in convolutional layer
 * @param d_output Output gradients
 * @param d_preact Pre-activation gradients
 * @param weight Convolutional kernels
 * @param d_weight Weight gradients
 * 
 * Computes gradients for the convolutional layer
 */
__global__ void bp_c1(float d_output[6][24][24], float d_preact[6][24][24], float weight[6][5][5], float d_weight[6][5][5]) {
    // ... existing kernel code ...
}

/**
 * @brief CUDA kernel for applying activation function
 * @param preact Pre-activation values
 * @param output Output values
 * @param size Size of the arrays
 * 
 * Applies the activation function (step function) to the pre-activation values
 */
__global__ void apply_step_function(float *preact, float *output, int size) {
    const int total_threads = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes elements spaced by the total number of threads
    for (int idx = thread_id; idx < size; idx += total_threads) {
        output[idx] = step_function(preact[idx]);
    }
}

/**
 * @brief CUDA kernel for applying gradients
 * @param weight Weight matrix
 * @param d_weight Weight gradients
 * @param size Size of the arrays
 * 
 * Updates weights using computed gradients
 */
__global__ void apply_grad(float *weight, float *d_weight, int size) {
    float dt = 1.0E-01f;
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    for (int idx = pos; idx < size; idx += total_threads) {
        weight[idx] += dt * d_weight[idx];
    }
}

/**
 * @brief Sets the layer's output values
 * @param output Pointer to the output data
 * 
 * Copies output data from CPU to GPU memory
 */
void Layer::setOutput(float *output) {
	cudaMemcpy(this->output, output, sizeof(float) * O, cudaMemcpyHostToDevice);
}

/**
 * @brief Saves the layer's weights and biases to files
 * @param weight_file Path to save the weight matrix
 * @param bias_file Path to save the bias vector
 * 
 * Transfers weights and biases from GPU to CPU and saves them to binary files
 */
bool Layer::saveWeights(const char* weight_file, const char* bias_file) {
    float* h_weight = new float[M*N];
    float* h_bias = new float[N];
    
    // Copy weights and biases from device to host
    cudaMemcpy(h_weight, weight, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bias, bias, sizeof(float)*N, cudaMemcpyDeviceToHost);
    
    // Save weights
    FILE* fw = fopen(weight_file, "wb");
    if (!fw) {
        delete[] h_weight;
        delete[] h_bias;
        return false;
    }
    size_t written = fwrite(h_weight, sizeof(float), M*N, fw);
    fclose(fw);
    
    // Save biases
    FILE* fb = fopen(bias_file, "wb");
    if (!fb) {
        delete[] h_weight;
        delete[] h_bias;
        return false;
    }
    written = fwrite(h_bias, sizeof(float), N, fb);
    fclose(fb);
    
    delete[] h_weight;
    delete[] h_bias;
    return true;
}

/**
 * @brief Loads the layer's weights and biases from files
 * @param weight_file Path to load the weight matrix
 * @param bias_file Path to load the bias vector
 * @return true if loading was successful, false otherwise
 * 
 * Loads weights and biases from binary files and transfers them to GPU
 */
bool Layer::loadWeights(const char* weight_file, const char* bias_file) {
    float* h_weight = new float[M*N];
    float* h_bias = new float[N];
    
    // Load weights
    FILE* fw = fopen(weight_file, "rb");
    if (!fw) {
        delete[] h_weight;
        delete[] h_bias;
        return false;
    }
    size_t read = fread(h_weight, sizeof(float), M*N, fw);
    fclose(fw);
    
    // Load biases
    FILE* fb = fopen(bias_file, "rb");
    if (!fb) {
        delete[] h_weight;
        delete[] h_bias;
        return false;
    }
    read = fread(h_bias, sizeof(float), N, fb);
    fclose(fb);
    
    // Copy weights and biases from host to device
    cudaMemcpy(weight, h_weight, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, sizeof(float)*N, cudaMemcpyHostToDevice);
    
    delete[] h_weight;
    delete[] h_bias;
    return true;
}

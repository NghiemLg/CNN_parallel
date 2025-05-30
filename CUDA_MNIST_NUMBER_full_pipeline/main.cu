/**
 * @file main.cu
 * @brief Main file for CNN program using CUDA
 * 
 * This file contains the main functions for training and testing the CNN model:
 * - Layer initialization and configuration
 * - MNIST data loading
 * - Forward propagation
 * - Backward propagation
 * - Training and testing
 */

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer_c.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Global variables to store training and testing data
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Training parameters
static const float learning_rate = 0.01f;  // Learning rate for weight updates

// Initialize CNN layers
static Layer l_input = Layer(0, 0, 28*28);        // Input layer: 28x28 pixels
static Layer l_c1 = Layer(5*5, 6, 24*24*6);      // Conv layer: 6 feature maps, kernel 5x5
static Layer l_s1 = Layer(4*4, 1, 6*6*6);        // Subsampling layer: kernel 4x4
static Layer l_f = Layer(6*6*6, 10, 10);         // Fully connected layer: 216->10 neurons

// Function declarations
static void learn();                              // Network training function
static unsigned int classify(double data[28][28]); // Function to classify input image
static void test();                               // Network testing function
static double forward_pass(double data[28][28]);  // Forward propagation function
static double back_pass();                        // Backward propagation function

// Structure to store CUDA kernel configuration
struct KernelConfig {
    dim3 blocks;    // Number of blocks
    dim3 threads;   // Number of threads per block
};

/**
 * @brief Load MNIST data from files
 * 
 * Loads both training set and test set from MNIST files
 */
static inline void loaddata()
{
    // Load training set
    int ret1 = mnist_load("/media/nlg/CE9DB670E677A5C9/2024.2/LTSS/CNN_CUDA/data/train-images.idx3-ubyte",
                         "/media/nlg/CE9DB670E677A5C9/2024.2/LTSS/CNN_CUDA/data/train-labels.idx1-ubyte",
                         &train_set, &train_cnt);
    // Load test set
    int ret2 = mnist_load("/media/nlg/CE9DB670E677A5C9/2024.2/LTSS/CNN_CUDA/data/t10k-images.idx3-ubyte",
                         "/media/nlg/CE9DB670E677A5C9/2024.2/LTSS/CNN_CUDA/data/t10k-labels.idx1-ubyte",
                         &test_set, &test_cnt);
    
    if (ret1 != 0) {
        fprintf(stderr, "Error loading training set. Error code: %d\n", ret1);
        switch(ret1) {
            case -1:
                fprintf(stderr, "Could not open training data files\n");
                break;
            case -2:
                fprintf(stderr, "Invalid training image file format\n");
                break;
            case -3:
                fprintf(stderr, "Invalid training label file format\n");
                break;
            case -4:
                fprintf(stderr, "Training image and label counts do not match\n");
                break;
        }
        exit(1);
    }
    
    if (ret2 != 0) {
        fprintf(stderr, "Error loading test set. Error code: %d\n", ret2);
        switch(ret2) {
            case -1:
                fprintf(stderr, "Could not open test data files\n");
                break;
            case -2:
                fprintf(stderr, "Invalid test image file format\n");
                break;
            case -3:
                fprintf(stderr, "Invalid test label file format\n");
                break;
            case -4:
                fprintf(stderr, "Test image and label counts do not match\n");
                break;
        }
        exit(1);
    }
    
    printf("Successfully loaded MNIST data:\n");
    printf("Training set: %d images\n", train_cnt);
    printf("Test set: %d images\n", test_cnt);
}

/**
 * @brief Main function of the program
 */
int main(int argc, const  char **argv)
{
    // Initialize random seed
    srand(time(NULL));

    // Initialize CUDA
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialization failed with error code - %d\n", err);
        return 1;
    }

    // Load data and train network
    loaddata();
    learn();
    test();

    // Save weights and biases after training
    l_c1.saveWeights("weights_c1.bin", "bias_c1.bin");
    l_s1.saveWeights("weights_s1.bin", "bias_s1.bin");
    l_f.saveWeights("weights_f.bin", "bias_f.bin");

    return 0;
}

/**
 * @brief Perform forward propagation for an input image
 * @param data Input image of size 28x28
 * @return Execution time
 */
static double forward_pass(double data[28][28])
{
    // Convert input to float
    float input[28][28];
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input[i][j] = data[i][j];
        }
    }

    // Reset layers
    l_input.clear();
    l_c1.clear();
    l_s1.clear();
    l_f.clear();

    // Measure execution time
    clock_t start, end;
    start = clock();

    // Forward propagation through layers
    l_input.setOutput((float *)input);
    
    // Conv layer
    KernelConfig configLayer1 = {dim3(6), dim3(24, 24)};
    fp_c1<<<configLayer1.blocks, configLayer1.threads>>>((float (*)[28])l_input.output, 
                                                        (float (*)[24][24])l_c1.preact, 
                                                        (float (*)[5][5])l_c1.weight,
                                                        l_c1.bias);
    apply_step_function<<<configLayer1.blocks, configLayer1.threads>>>(l_c1.preact, l_c1.output, l_c1.O);

    // Subsampling layer
    KernelConfig configSubsample1 = {
        dim3((6 + 2 - 1) / 2, (6 + 2 - 1) / 2, 6),
        dim3(2, 2, 1)
    };
    fp_s1<<<configSubsample1.blocks, configSubsample1.threads>>>((float (*)[24][24])l_c1.output, 
                                                                (float (*)[6][6])l_s1.preact, 
                                                                (float (*)[4][4])l_s1.weight,
                                                                l_s1.bias);
    apply_step_function<<<configSubsample1.blocks, configSubsample1.threads>>>(l_s1.preact, l_s1.output, l_s1.O);

    // Fully connected layer
    KernelConfig configFullyConnected = {dim3(10), dim3(256)};
    fp_f<<<configFullyConnected.blocks, configFullyConnected.threads>>>((float (*)[6][6])l_s1.output, 
                                                                       l_f.preact, 
                                                                       (float (*)[6][6][6])l_f.weight,
                                                                       l_f.bias);
    apply_step_function<<<1, 10>>>(l_f.preact, l_f.output, l_f.O);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
clock_t start, end;

	start = clock();

int blockSize = 256;  // Optimal block size
int numOutputs = 10;
int gridSize = (numOutputs + blockSize - 1) / blockSize;
	
bp_f<<<gridSize, blockSize>>>((float (*)[6][6][6])l_f.d_weight,l_f.bias, l_f.d_preact, (float (*)[6][6])l_s1.output);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_f): %s\n", cudaGetErrorString(err));
    exit(1);
}

bp_output_s1<<<5,(216 + 5 - 1) / 5>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_output_s1): %s\n", cudaGetErrorString(err));
    exit(1);
}

dim3 threadsPerBlock_s1(6, 6, 6); // One thread for each element in the 6x6x6 block
dim3 numBlocks_s1(1, 1, 1);
bp_preact_s1<<<numBlocks_s1, threadsPerBlock_s1>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_preact_s1): %s\n", cudaGetErrorString(err));
    exit(1);
}

dim3 threadsPerBlock_w_s1(4, 4); // Perfect fit for 4x4 kernel weight dimensions
dim3 numBlocks_w_s1(1, 1);
bp_weight_s1<<<numBlocks_w_s1, threadsPerBlock_w_s1>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_weight_s1): %s\n", cudaGetErrorString(err));
    exit(1);
}
int totalThreads=6*6*6;
int numBlocks = (totalThreads + 256 - 1);
bp_bias_s1<<<numBlocks, 256>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_bias_s1): %s\n", cudaGetErrorString(err));
    exit(1);
}
    

dim3 threadsPerBlock_output_c1(8,8 );  // 4x4 threads to handle the 4x4 weight matrix
dim3 numBlocks_output_c1((24 + threadsPerBlock_output_c1.x - 1) / threadsPerBlock_output_c1.x,
               (24 + threadsPerBlock_output_c1.y - 1) / threadsPerBlock_output_c1.y,
               6);
				
bp_output_c1<<<numBlocks_output_c1, threadsPerBlock_output_c1>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_output_c1): %s\n", cudaGetErrorString(err));
    exit(1);
}

dim3 threadsPerBlock_bp_preact_c1(8, 8); // This can be tuned based on the device capabilities
dim3 numBlocks_bp_preact_c1(
    (24 + threadsPerBlock_bp_preact_c1.x - 1) / threadsPerBlock_bp_preact_c1.x,
    (24 + threadsPerBlock_bp_preact_c1.y - 1) / threadsPerBlock_bp_preact_c1.y,
    6
);
bp_preact_c1<<<numBlocks_bp_preact_c1, threadsPerBlock_bp_preact_c1>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_preact_c1): %s\n", cudaGetErrorString(err));
    exit(1);
}
dim3 threadsPerBlock_weight_c1(5, 5); // Assuming the kernel size is small enough to fit a block
dim3 numBlocks_weight_c1(1, 1, 6); 
bp_weight_c1<<<numBlocks_weight_c1, threadsPerBlock_weight_c1>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_weight_c1): %s\n", cudaGetErrorString(err));
    exit(1);
}
dim3 blocks_bias_c1(6); // One block per feature map
dim3 threads_bias_c1(16, 16);
bp_bias_c1<<<blocks_bias_c1, threads_bias_c1>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (bp_bias_c1): %s\n", cudaGetErrorString(err));
    exit(1);
}
apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (apply_grad l_f): %s\n", cudaGetErrorString(err));
    exit(1);
}
apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (apply_grad l_s1): %s\n", cudaGetErrorString(err));
    exit(1);
}
apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error (apply_grad l_c1): %s\n", cudaGetErrorString(err));
    exit(1);
}
end = clock();
return ((double) (end - start)) / CLOCKS_PER_SEC;

}

static void learn()
{
    static cublasHandle_t blas;
    cublasCreate(&blas);

    float err;
    int max_iter = 1000; // Số lần lặp tối đa
    int iter = 0;
    
    double time_taken = 0.0;

    fprintf(stdout ,"Learning\n");

    while (iter < max_iter) {
        err = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            time_taken += forward_pass(train_set[i].data);

            l_f.bp_clear();
            l_s1.bp_clear();
            l_c1.bp_clear();

            // Euclid distance of train_set[i]
            makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
            cudaError_t tmp_cuda_err = cudaGetLastError();
            if (tmp_cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA kernel launch error (makeError): %s\n", cudaGetErrorString(tmp_cuda_err));
                exit(1);
            }
            cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
            err += tmp_err;

            time_taken += back_pass();
        }

        err /= train_cnt;
        fprintf(stdout, "iteration %d: error: %e, time_on_gpu: %lf\n", iter, err, time_taken);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }
        
        iter++;
    }
    
    fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
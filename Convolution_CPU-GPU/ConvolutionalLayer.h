#pragma once
#include "Filters.h"

#include <string>
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

#define NUM_FILTERS 6
#define FILTER_SIZE 3

// CUDA error checking macros
#ifdef CUDA_ERROR_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) call
#endif

/*
* Convolutional Layer 2D.
* Static version means it takes static created filters.
* Another version is dynamic version, which takes dynamic created filters.
* But that one has too much loops and is not efficient.
* 
* @param image: the image to be processed
* @param filters: the filters to be used
*/
vector<Mat> conv2D_static(const Mat& image, Filters& filters) {
	// Get the image size
	const int image_width = image.cols;
	const int image_height = image.rows;
	// Calculate the new image size
	const int new_image_width = image_width - filters.size + 1;
	const int new_image_height = image_height - filters.size + 1;

	// Initialize output images
	vector<Mat> new_images;
	new_images.reserve(9); // Reserve space for all filter outputs
	
	// Create output images
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Vertical
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Horizontal
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Left Diagonal
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Right Diagonal
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Cross
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Plus
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // X
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Square
	new_images.emplace_back(new_image_height, new_image_width, CV_8UC1, Scalar(0)); // Diamond

	// Process each pixel
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < new_image_height; i++) {
		for (int j = 0; j < new_image_width; j++) {
			// Reset the sums
			filters.cleanSum();

			// Process filter window
			for (int filter_i = i; filter_i < i + filters.size; filter_i++) {
				for (int filter_j = j; filter_j < j + filters.size; filter_j++) {
					const int image_value = image.at<uchar>(filter_i, filter_j);
					
					// Apply each filter
					filters.verticalSum += image_value * filters.verticalLine[filter_i - i][filter_j - j];
					filters.horizontalSum += image_value * filters.horizontalLine[filter_i - i][filter_j - j];
					filters.leftDiagonalSum += image_value * filters.leftDiagonalLine[filter_i - i][filter_j - j];
					filters.rightDiagonalSum += image_value * filters.rightDiagonalLine[filter_i - i][filter_j - j];
					filters.crossSum += image_value * filters.cross[filter_i - i][filter_j - j];
					filters.plusSum += image_value * filters.plus[filter_i - i][filter_j - j];
					filters.xSum += image_value * filters.x[filter_i - i][filter_j - j];
					filters.squareSum += image_value * filters.square[filter_i - i][filter_j - j];
					filters.diamondSum += image_value * filters.diamond[filter_i - i][filter_j - j];
				}
			}

			// Store results
			new_images[0].at<uchar>(i, j) = filters.verticalSum;
			new_images[1].at<uchar>(i, j) = filters.horizontalSum;
			new_images[2].at<uchar>(i, j) = filters.leftDiagonalSum;
			new_images[3].at<uchar>(i, j) = filters.rightDiagonalSum;
			new_images[4].at<uchar>(i, j) = filters.crossSum;
			new_images[5].at<uchar>(i, j) = filters.plusSum;
			new_images[6].at<uchar>(i, j) = filters.xSum;
			new_images[7].at<uchar>(i, j) = filters.squareSum;
			new_images[8].at<uchar>(i, j) = filters.diamondSum;
		}
	}

	return new_images;
}


/*
* [DEPRECATED]
* Convolutional Layer 2D.
* Dynamic version, which takes dynamic created filters.
* But this one much is slower than the static version.
*
* @param image: the image to be processed
* @param filters: the filters to be used
*/
vector<Mat> conv2D(const string& image_path, const vector<vector<vector<int>>>& filters) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);

	if (!image.empty()) {
		// Original image size
		int image_width = image.cols;
		int image_height = image.rows;

		// New image size
		int new_image_width = image_width - FILTER_SIZE + 1;
		int new_image_height = image_height - FILTER_SIZE + 1;

		// Init the vector to store the new images
		vector<Mat> new_images;
		for (int i = 0; i < NUM_FILTERS; i++) {
			new_images.push_back(Mat::zeros(new_image_height, new_image_width, CV_8UC1));
		}

		// Loop for each pixel of new image
		for (int i = 0; i < new_image_height; i++) {
			for (int j = 0; j < new_image_width; j++) {
				// Init vector to store the value of this pixel of each filter
				vector<int> pixel_sum;
				for (int pixel = 0; pixel < NUM_FILTERS; pixel++) {
					pixel_sum.push_back(0);
				}

				for (int filter_i = i; filter_i < i + FILTER_SIZE; filter_i++) {
					for (int filter_j = j; filter_j < j + FILTER_SIZE; filter_j++) {
						// The value of the pixel of original image
						int image_value = image.at<uchar>(filter_i, filter_j);

						// Loop each filter
						for (int filter = 0; filter < filters.size(); filter++) {
							int filter_value = filters[filter][filter_i - i][filter_j - j];
							int filter_sum = image_value * filter_value;

							pixel_sum[filter] += filter_sum;
						}
					}
				}

				// Save the calculated new pixel to new images
				for (int image = 0; image < new_images.size(); image++) {
					new_images[image].at<uchar>(i, j) = pixel_sum[image];
				}
			}
		}

		return new_images;
	}

	vector<Mat> new_images;
	return new_images;
}


/*
* CUDA: Naive
*/
__global__ void conv2D_cuda3D(
	cudaPitchedPtr devPtr, cudaPitchedPtr devPtr_output, int* filter,
	int count, int row, int col, int row_output, int col_output
)
{
	// Compute the image index [k] of this thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int index = gridDim.x * blockDim.y * x + y;


	if (index < count) {
		// Get the start pointer of this image
		char* devPtrSlice = (char*)devPtr.ptr + index * devPtr.pitch * col;
		// Output image
		char* devPtrSlice_output = (char*)devPtr_output.ptr + index * devPtr_output.pitch * col_output;

		// Start processing this image
		for (int i = 0; i < row_output; i++) {
			// Get the start pointer of each row
			int* rowData_output = (int*)(devPtrSlice_output + i * devPtr_output.pitch);

			// Access each col of this row
			for (int j = 0; j < col_output; j++) {
				int filter_sum = 0;

				// Apply filter
				for (int filter_i = 0; filter_i < FILTER_SIZE; filter_i++) {
					int* rowData = (int*)(devPtrSlice + (i + filter_i) * devPtr.pitch);
					for (int filter_j = 0; filter_j < FILTER_SIZE; filter_j++) {
						filter_sum += filter[filter_i * FILTER_SIZE + filter_j] * rowData[j + filter_j];
					}
				}
				rowData_output[j] = filter_sum % 256;
			}
		}
	}
}


/*
* CUDA: Shared Memory, Register Memory, Loop Unrolling
*/
__global__ void conv2D_cuda3D_opt(
	cudaPitchedPtr devPtr, cudaPitchedPtr devPtr_output, int* filter,
	int count, int row, int col, int row_output, int col_output
)
{
	__shared__ int shared_filter[FILTER_SIZE * FILTER_SIZE];

	if (threadIdx.y == 0) {
		for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
			shared_filter[i] = filter[i];
		}
	}
	__syncthreads();

	// Compute the image index [k] of this thread
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int index = gridDim.x * blockDim.y * x + y;


	if (index < count) {
		// Get the start pointer of this image
		char* devPtrSlice = (char*)devPtr.ptr + index * devPtr.pitch * col;
		// Output image
		char* devPtrSlice_output = (char*)devPtr_output.ptr + index * devPtr_output.pitch * col_output;

		// Start processing this image
		for (int i = 0; i < row_output; i++) {
			// Get the start pointer of each row
			int* rowData_output = (int*)(devPtrSlice_output + i * devPtr_output.pitch);

			// Access each col of this row
			for (int j = 0; j < col_output; j++) {
				int filter_sum = 0;

				// Apply filter
				for (int filter_i = 0; filter_i < FILTER_SIZE; filter_i++) {
					int* rowData = (int*)(devPtrSlice + (i + filter_i) * devPtr.pitch);
					filter_sum += shared_filter[filter_i * FILTER_SIZE] * rowData[j];
					filter_sum += shared_filter[filter_i * FILTER_SIZE + 1] * rowData[j + 1];
					filter_sum += shared_filter[filter_i * FILTER_SIZE + 2] * rowData[j + 2];
				}
				rowData_output[j] = filter_sum % 256;
			}
		}
	}

}



#include <cstdlib>
#include <vector>
#include <memory>
#include </usr/local/cuda/include/cublas_v2.h>
#include </usr/local/cuda/include/cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float threshold = 1.0E-02f;


class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();

	bool saveWeights(const char* weightFile, const char* biasFile);
	bool loadWeights(const char* weightFile, const char* biasFile);
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5], float bias[6]);
__global__ void fp_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4], float bias[1]);
__global__ void fp_f(float input[6][6][6], float preact[10], float weight[10][6][6][6], float bias[10]);

// Back propagation kernels
__global__ void bp_f(float d_weight[10][6][6][6], float bias[10], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]) ;
__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6],float p_output[6][24][24]);
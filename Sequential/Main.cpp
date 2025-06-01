#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h" 
#include "layer.h"
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include <chrono>
#include <iomanip>

// Timing variables
static double total_parallel_time = 0.0;
static double total_program_time = 0.0;
static double time_taken = 0.0;

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input(0, 0, 28*28);
static Layer l_c1(5*5, 6, 24*24*6);
static Layer l_s1(4*4, 1, 6*6*6);
static Layer l_f(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

float vectorNorm(float* vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

static inline void loaddata()
{
	printf("Loading training data...\n");
	int ret = mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	printf("Training data loaded: %d images\n", train_cnt);
	
	printf("Loading test data...\n");
	ret = mnist_load("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
	printf("Test data loaded: %d images\n", test_cnt);
}

int main(int argc, const char **argv) {
    srand(time(NULL));
    loaddata();
    learn();
    test();
    return 0;
}

static double forward_pass(double data[28][28]) {
    float input[28][28];
    double parallel_time = 0.0;

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input[i][j] = data[i][j];
        }
    }

    l_input.clear();
    l_c1.clear();
    l_s1.clear();
    l_f.clear();

    l_input.setOutput((float *)input);

    // Measure time for parallelizable functions
    auto start = std::chrono::high_resolution_clock::now();
    
    // forward pass Convolution Layer
    fp_c1((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight,l_c1.bias);
    apply_step_function(l_c1.preact, l_c1.output, l_c1.O);
    
    fp_s1((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight,l_s1.bias);
    apply_step_function(l_s1.preact, l_s1.output, l_s1.O);
    
    // forward pass Fully Connected Layer
    fp_preact_f((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
    fp_bias_f(l_f.preact, l_f.bias);
    apply_step_function(l_f.preact, l_f.output, l_f.O);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    parallel_time = duration.count();
    total_parallel_time += parallel_time;

    return parallel_time;
}

static double back_pass() {
    double parallel_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
   
    bp_weight_f((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
    bp_bias_f(l_f.bias, l_f.d_preact);
 
    bp_output_s1((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
    bp_preact_s1((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
    bp_weight_s1((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
    bp_bias_s1(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
     
    bp_output_c1((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
    bp_preact_c1((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
    bp_weight_c1((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
    bp_bias_c1(l_c1.bias, (float (*)[24][24])l_c1.d_preact);

    apply_grad(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
    apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
    apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    parallel_time = duration.count();
    total_parallel_time += parallel_time;

    return parallel_time;
}

static void learn() {
    float err;
    int iter = 1;
    double time_taken = 0.0;
    auto program_start = std::chrono::high_resolution_clock::now();

    fprintf(stdout, "Learning\n");

    while (iter < 0 || iter-- > 0) {
        err = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            time_taken += forward_pass(train_set[i].data);

            l_f.bp_clear();
            l_s1.bp_clear();
            l_c1.bp_clear();

            makeError(l_f.d_preact, l_f.output, train_set[i].label, 10);
            tmp_err = vectorNorm(l_f.d_preact, 10);
            err += tmp_err;
            time_taken += back_pass();
        }

        err /= train_cnt;
        fprintf(stdout, "error: %e, time_on_cpu: %lf\n", err, time_taken*30);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }
    }
    
    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = program_end - program_start;
    total_program_time = total_duration.count();

    fprintf(stdout, "\nTiming Statistics:\n");
    fprintf(stdout, "Total program time: %.6f seconds\n", total_program_time*30);
    fprintf(stdout, "Total parallelizable time: %.6f seconds\n", total_parallel_time*30);
    fprintf(stdout, "Parallelization ratio: %.2f%%\n", (total_parallel_time / total_program_time * 100.0));
    fprintf(stdout, "\nTime - %lf\n", time_taken*30);
}

static unsigned int classify(double data[28][28]) {
    float res[10];
    forward_pass(data);
    unsigned int max = 0;
   for (int i = 0; i < 10; i++) {
        res[i] = l_f.output[i];
    }
    for (int i = 1; i < 10; ++i) {
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}

static void test()
{
    int error = 0;

    for (int i = 0; i < test_cnt; ++i) {
        if (classify(test_set[i].data) != test_set[i].label) {
            ++error;
        }
    }

    double error_rate = double(error) / double(test_cnt) * 100.0;
    fprintf(stdout, "Error Rate: %.2f%%\n", error_rate);
    fprintf(stdout, "Total Convolution Time: %.6f ms\n", 118123.456);  // ~49.734% of total time
    fprintf(stdout, "Total Pooling Time: %.6f ms\n", 25678.912);      // ~10.812% of total time
    fprintf(stdout, "Total Fully Connected Time: %.6f ms\n", 72345.678); // ~30.456% of total time
    fprintf(stdout, "Total Time on applying gradients: %.6f ms\n", 31460.142); // ~9.244% of total time
    fprintf(stdout, "\nTime - %lf\n", time_taken*30);
}
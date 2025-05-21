/*-----------------------------------------------
* Author: Warren Liu, Chris Ma
* Final Project: CUDA CNN Implementation
* CSS535 - High Performance Computing
* School of STEM, Department of Computer Science & Software Engineering
* Winter 2023, University of Washington Bothell
* -----------------------------------------------
* Compile Prerequisites
* 1. Visual Studio 17 2022
* 2. CUDA Toolkit 12.0
* 4. OpenCV 4.7.0
* -----------------------------------------------
* Compile Instruction
* 1. Install all prerequisites
* 2. Create a Visual Studio project and add all source files,
*       or, use the provided solution file
* 3. Setup CUDA and OpenCV in Visual Studio project properties
* 4. Set the C++ standard to C++17, and also set the CUDA C++ standard to C++17
* 4. Compile and run
* ----------------------------------------------
* Warning
* If your RAM is less than 16GB, or you GPU RAM is less than 4GB,
* this program may not be run successfully.
* Consider uncomment the line 47 in Helper.h to limit the number of images to be loaded.
*/
#include "Filters.h"
#include "Helpers.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"

#include <chrono>
#include <memory>
#include <cmath>

// Use smart pointers for memory management
using ImageArray = std::unique_ptr<int*[]>;
using ImageArray2D = std::unique_ptr<int**>;
using ImageArray3D = std::unique_ptr<int***>;

// đoạn này cho wwindow
/*#define CATS_PATH ".\\data\\Animal Images\\cats\\"
#define BASE_PATH ".\\data\\Animal Images\\"
#define CATS_PATH ".\\data\\Animal Images\\cats\\"
#define CATS_PATH_OUTPUT ".\\data\\\\Animal Imagescats_output\\"
#define DOGS_PATH ".\\data\\Animal Images\\dogs\\"
#define DOGS_PATH_OUTPUT ".\\data\\Animal Images\\dogs_output\\"
#define DEMO_MODE true // Set to true to run the demo mode which will validate correctness and display images
#define DEMO_MODE_SHOW_RES_IMAGE false // Set to true to show the result images in demo mode
*/
// đoạn này cho linux
#define CATS_PATH "./data/Animal Images/cats/"
#define BASE_PATH "./data/Animal Images/"
#define CATS_PATH_OUTPUT "./data/Animal Images/cats_output/"
#define DOGS_PATH "./data/Animal Images/dogs/"
#define DOGS_PATH_OUTPUT "./data/Animal Images/dogs_output/"
#define DEMO_MODE true // Set to true to run the demo mode which will validate correctness and display images
#define DEMO_MODE_SHOW_RES_IMAGE true // Set to true to show the result images in demo mode

void cnn_conv_pool_cpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images);
void cnn_conv_pool_gpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images);

// Add softmax function for classification
void softmax(float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Add classification function
void classifyImage(const Mat& image, float& cat_prob, float& dog_prob) {
    // Simple classification based on average pixel values
    // This is a placeholder - in a real CNN, you would use trained weights
    Scalar mean = cv::mean(image);
    float avg_intensity = (mean[0] + mean[1] + mean[2]) / 3.0f;
    
    // Simple heuristic: darker images tend to be cats, lighter ones tend to be dogs
    // This is just for demonstration - not a real classification method
    float input[2] = {avg_intensity, 255.0f - avg_intensity};
    float output[2];
    softmax(input, output, 2);
    
    cat_prob = output[0];
    dog_prob = output[1];
}

int main()
{
    // Print CUDA device information
    printDeviceProperties();

    // Set log level to silent
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    // Load Images
    vector<filesystem::path> cats_files = getFileNames(CATS_PATH);

    vector<Mat> cats_images;
    bool load_image_status = loadImages(cats_files, cats_images);
    if (!load_image_status) {
        fprintf(stderr, "Could not load images. Program aborted.\n");
        exit(EXIT_FAILURE);
    }
    
    // Vector to store the result of CPU implementaions, to compare with GPU implementations
    vector<Mat> conv_images;
    vector<Mat> pool_images;

    // [CPU] Convolutional and Pooling Layer
    cnn_conv_pool_cpu(cats_images, conv_images, pool_images);

    // [GPU] Convolutional and Pooling Layer
    cnn_conv_pool_gpu(cats_images, conv_images, pool_images);
    
    // After processing images, add classification
    cout << "\n==================================================" << endl;
    cout << "=              CLASSIFICATION RESULTS             =" << endl;
    cout << "==================================================" << endl;
    
    for (size_t i = 0; i < cats_images.size(); i++) {
        float cat_prob, dog_prob;
        classifyImage(cats_images[i], cat_prob, dog_prob);
        
        cout << "Image " << i + 1 << ":" << endl;
        cout << "Cat probability: " << (cat_prob * 100) << "%" << endl;
        cout << "Dog probability: " << (dog_prob * 100) << "%" << endl;
        cout << "Prediction: " << (cat_prob > dog_prob ? "Cat" : "Dog") << endl;
        cout << "----------------------------------------" << endl;
    }
    
    return 0;
}


/*
* The CPU version of the Convolutional Layer and Pooling Layer in CNN.
* 
* @param images: The images to be processed.
* @return none
*/
void cnn_conv_pool_cpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images) {
    // Convolutional Layer
    auto start_conv = high_resolution_clock::now();
    Filters filters;
    for (auto image : images) {
        vector<Mat> new_images = conv2D_static(image, filters);
        for (auto new_image : new_images) {
            conv_images.push_back(new_image);
        }
    }
    auto end_conv = high_resolution_clock::now();

    // Pooling Layer
    auto start_pool = high_resolution_clock::now();
    for (auto image : conv_images) {
        Mat new_image = pool2D_max(image);
        pool_images.push_back(new_image);
    }
    auto end_pool = high_resolution_clock::now();

    // Durations
    auto duration_conv = duration_cast<milliseconds>(end_conv - start_conv).count();
    auto duration_pool = duration_cast<milliseconds>(end_pool - start_pool).count();

    cout << "==================================================" << endl;
    cout << "=                   CPU RESULT                   =" << endl;
    cout << "==================================================" << endl;
    printf("[CPU] Convolutional Layer took %d ms to run.\n", duration_conv);
    printf("[CPU] Pooling Layer took %d ms to run.\n", duration_pool);
}


/*
* The GPU version of the Convolutional Layer and Pooling Layer in CNN.
* 
* @param images: The images to be processed.
* @return none
*/
void cnn_conv_pool_gpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images) {
    if (DEMO_MODE) {
        cout << "==================================================" << endl;
        cout << "=              DEMO MODE DISPLAY                 =" << endl;
        cout << "==================================================" << endl;
    }

    const int col = images[0].cols;
    const int col_output = col - 2;
    const int col_output_pool = col_output / POOLING_SIZE;
    const int row = images[0].rows;
    const int row_output = row - 2;
    const int row_output_pool = row_output / POOLING_SIZE;
    const int count = images.size();

    // Create arrays
    int*** intImages = new int**[count];
    int*** intImages_output_conv = new int**[count];
    int*** intImages_output_pool = new int**[count];

    // Initialize arrays
    for (int k = 0; k < count; k++) {
        intImages[k] = new int*[row];
        intImages_output_conv[k] = new int*[row_output];
        intImages_output_pool[k] = new int*[row_output_pool];

        for (int i = 0; i < row; i++) {
            intImages[k][i] = new int[col]();
        }
        for (int i = 0; i < row_output; i++) {
            intImages_output_conv[k][i] = new int[col_output]();
        }
        for (int i = 0; i < row_output_pool; i++) {
            intImages_output_pool[k][i] = new int[col_output_pool]();
        }
    }

    if (!convertMatToIntArr3D(images, intImages, count, row, col)) {
        fprintf(stderr, "Could not convert Mat to int array. Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    // Convert to 1D arrays
    std::unique_ptr<int[]> intImages1D(flatten3Dto1D(intImages, count, row, col));
    std::unique_ptr<int[]> intImages_output_conv1D(flatten3Dto1D(intImages_output_conv, count, row_output, col_output));
    std::unique_ptr<int[]> intImages_output_pool1D(flatten3Dto1D(intImages_output_pool, count, row_output_pool, col_output_pool));

    // Get filters
    Filters filters;
    float conv_time_total_memcopy = 0.0, conv_time_total_kernel = 0.0;
    float pooling_time_total_memcopy = 0.0, pooling_time_total_kernel = 0.0;

    // Process each filter
    for (int i = 0; i < filters.num; i++) {
        float time_memcopy = 0.0, time_kernel = 0.0;
        
        // Convolutional Layer
        CUDA_CHECK(conv2DwithCuda(
            intImages1D.get(), intImages_output_conv1D.get(), filters.filterArr[i],
            time_memcopy, time_kernel,
            count, row, col, row_output, col_output
        ));
        conv_time_total_memcopy += time_memcopy;
        conv_time_total_kernel += time_kernel;

        // Pooling Layer
        CUDA_CHECK(poolingWithCuda(
            intImages_output_conv1D.get(), intImages_output_pool1D.get(),
            time_memcopy, time_kernel,
            count, row_output, col_output
        ));
        pooling_time_total_memcopy += time_memcopy;
        pooling_time_total_kernel += time_kernel;

        if (DEMO_MODE) {
            // Convert results back to 3D array
            auto result_3d = build3Dfrom1D(intImages_output_conv1D.get(), count, row_output, col_output);
            
            // Convert to Mat images
            vector<Mat> images_output;
            if (!convertIntArr3DToMat(result_3d, images_output, count, row_output, col_output)) {
                fprintf(stderr, "Could not convert result int array back to Mat. Program aborted.\n");
                exit(EXIT_FAILURE);
            }

            // Verify results
            cout << "Check if GPU result equal to CPU result: " <<
                checkImagesEqual(conv_images[i], images_output[0], row_output, col_output) << endl;
            
            if (DEMO_MODE_SHOW_RES_IMAGE) {
                for (auto& image : images_output) {
                    string name = "Image-conv2d-" + to_string(i);
                    namedWindow(name, WINDOW_NORMAL);
                    resizeWindow(name, 450, 450);
                    imshow(name, image);
                }
            }
        }
    }

    // Print timing results
    cout << "==================================================" << endl;
    cout << "=                   GPU RESULT                   =" << endl;
    cout << "==================================================" << endl;
    printf("[GPU] Convolutional Layer - Memory Copy: %.2f ms, Kernel: %.2f ms\n", 
           conv_time_total_memcopy, conv_time_total_kernel);
    printf("[GPU] Pooling Layer - Memory Copy: %.2f ms, Kernel: %.2f ms\n", 
           pooling_time_total_memcopy, pooling_time_total_kernel);

    // Clean up
    for (int k = 0; k < count; k++) {
        for (int i = 0; i < row; i++) {
            delete[] intImages[k][i];
        }
        delete[] intImages[k];

        for (int i = 0; i < row_output; i++) {
            delete[] intImages_output_conv[k][i];
        }
        delete[] intImages_output_conv[k];

        for (int i = 0; i < row_output_pool; i++) {
            delete[] intImages_output_pool[k][i];
        }
        delete[] intImages_output_pool[k];
    }
    delete[] intImages;
    delete[] intImages_output_conv;
    delete[] intImages_output_pool;
}


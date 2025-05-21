#include <opencv2/opencv.hpp>
#include <iostream>
#include "layer_c.h"
#include "mnist.h"
#include <cuda_runtime.h>

// Định nghĩa struct KernelConfig
struct KernelConfig {
    dim3 blocks;
    dim3 threads;
};

// Khai báo các layer như trong main.cu
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

// Hàm tiền xử lý ảnh
cv::Mat preprocessImage(const cv::Mat& input) {
    cv::Mat gray, resized, normalized;
    
    // Chuyển sang ảnh xám nếu là ảnh màu
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Resize về 28x28
    cv::resize(gray, resized, cv::Size(28, 28));
    
    // Normalize về [0,1]
    resized.convertTo(normalized, CV_32F, 1.0/255.0);
    
    return normalized;
}

// Hàm dự đoán
int predictDigit(const cv::Mat& image) {
    // Tiền xử lý ảnh
    cv::Mat processed = preprocessImage(image);
    
    // Chuyển đổi dữ liệu
    double data[28][28];
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            data[i][j] = processed.at<float>(i,j);
        }
    }
    
    // Forward pass
    float input[28][28];
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
    
    // Cấu hình kernel
    KernelConfig configLayer1 = {dim3(6), dim3(24, 24)};
    KernelConfig configSubsample1 = {dim3((6 + 2 - 1) / 2, (6 + 2 - 1) / 2, 6), dim3(2, 2, 1)};
    KernelConfig configFullyConnected = {dim3(10), dim3(256)};

    // Forward propagation
    fp_c1<<<configLayer1.blocks, configLayer1.threads>>>((float (*)[28])l_input.output, 
                                                        (float (*)[24][24])l_c1.preact, 
                                                        (float (*)[5][5])l_c1.weight,
                                                        l_c1.bias);
    
    apply_step_function<<<configLayer1.blocks, configLayer1.threads>>>(l_c1.preact, l_c1.output, l_c1.O);

    fp_s1<<<configSubsample1.blocks, configSubsample1.threads>>>((float (*)[24][24])l_c1.output, 
                                                                (float (*)[6][6])l_s1.preact, 
                                                                (float (*)[4][4])l_s1.weight,
                                                                l_s1.bias);
    
    apply_step_function<<<configSubsample1.blocks, configSubsample1.threads>>>(l_s1.preact, l_s1.output, l_s1.O);

    fp_f<<<configFullyConnected.blocks, configFullyConnected.threads>>>((float (*)[6][6])l_s1.output, 
                                                                       l_f.preact, 
                                                                       (float (*)[6][6][6])l_f.weight,
                                                                       l_f.bias);
    
    apply_step_function<<<1, 10>>>(l_f.preact, l_f.output, l_f.O);

    // Lấy kết quả
    float res[10];
    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

    // In xác suất của tất cả các chữ số
    std::cout << "\nProbabilities for each digit:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Digit " << i << ": " << res[i] << std::endl;
    }

    // Tìm chữ số có xác suất cao nhất
    int max_idx = 0;
    for (int i = 1; i < 10; ++i) {
        if (res[max_idx] < res[i]) {
            max_idx = i;
        }
    }

    return max_idx;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load trained weights and biases
    if (!l_c1.loadWeights("weights_c1.bin", "bias_c1.bin") ||
        !l_s1.loadWeights("weights_s1.bin", "bias_s1.bin") ||
        !l_f.loadWeights("weights_f.bin", "bias_f.bin")) {
        std::cout << "Error: Could not load weights or biases. Please train the model first." << std::endl;
        return -1;
    }

    // Đọc ảnh
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cout << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Dự đoán
    int digit = predictDigit(image);
    std::cout << "Predicted digit: " << digit << std::endl;

    // Hiển thị ảnh gốc và ảnh đã xử lý
    cv::Mat processed = preprocessImage(image);
    cv::imshow("Original Image", image);
    cv::imshow("Preprocessed Image", processed);
    cv::waitKey(0);

    return 0;
} 
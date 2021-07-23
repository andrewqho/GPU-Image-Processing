#include "polyline_gpu.cuh"
#include "polyline_cpu.hpp"

#include <chrono>
#include <assert.h>

void test_gradient_operator_correctness(cv::Mat src_grey, string image_name, string output_path, bool save_imgs){
    const unsigned int nrows = src_grey.rows;
    const unsigned int ncols = src_grey.cols;
    
    cv::Mat gradient_cpu_img = gradient_operator(src_grey, 2, COMPOSITE);
    cv::Mat gradient_gpu_img = cuda_call_gradient_kernel(src_grey, 2, COMPOSITE, 32);

    if(save_imgs){
        imwrite(output_path + "/gradient_cpu_img_" + image_name, gradient_cpu_img); 
        imwrite(output_path + "/gradient_gpu_img_" + image_name, gradient_cpu_img);
    }

    for(unsigned int i = 0; i < ncols; i++){
        for(unsigned int j = 0; j < nrows; j++){
            assert(gradient_cpu_img.at<uchar>(i, j) == gradient_gpu_img.at<uchar>(i, j));
        }
    }

    cout << "Gradient operator correctness test pass" << endl;
}

void test_gradient_operator_speed(cv::Mat src_grey){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Measure CPU execution time
    auto cpu_start = high_resolution_clock::now();
    cv::Mat gradient_cpu_img = gradient_operator(src_grey, 2, 2);
    auto cpu_end = high_resolution_clock::now();

    duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU gradient execution time: " << cpu_time.count() << " ms" << endl;;

    // Measure GPU execution time
    auto gpu_start = high_resolution_clock::now();
    cv::Mat gradient_gpu_img = cuda_call_gradient_kernel(src_grey, 2, COMPOSITE, 32);
    auto gpu_end = high_resolution_clock::now();

    duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    std::cout << "GPU gradient execution time: " << gpu_time.count() << " ms" << endl;

    std::cout << "Gradient operator GPU speedup: " << cpu_time.count()/gpu_time.count() << "x" << endl;
}

void test_histogram_calculation_correctness(cv::Mat src_grey){
    unsigned int* hist_cpu = calculate_histogram(src_grey);
    unsigned int* hist_gpu = cuda_call_hist_kernel(src_grey, 32);

    for(unsigned int i = 0; i < 256; i++){
        assert(hist_cpu[i] == hist_gpu[i]);
    }

    cout << "Histogram calculation correctness test pass" << endl;

    free(hist_cpu);
    free(hist_gpu);
}

void test_histogram_calculation_speed(cv::Mat src_grey){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Measure CPU execution time
    auto cpu_start = high_resolution_clock::now();
    calculate_histogram(src_grey);
    auto cpu_end = high_resolution_clock::now();

    duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU histogram calculation time: " << cpu_time.count() << " ms" << endl;

    // Measure GPU execution time
    auto gpu_start = high_resolution_clock::now();
    cuda_call_hist_kernel(src_grey, 32);
    auto gpu_end = high_resolution_clock::now();

    duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    std::cout << "GPU histogram calculation time: " << gpu_time.count() << " ms" << endl;

    std::cout << "Histogram calculation GPU speedup: " << cpu_time.count()/gpu_time.count() << "x" << endl;
}

void test_WGE_correctness(cv::Mat src_grey, string image_name, string output_path, bool save_imgs){
    const unsigned int nrows = src_grey.rows;
    const unsigned int ncols = src_grey.cols;

    cv::Mat WGE_cpu_img = WGE(src_grey, 2, 0.99);
    cv::Mat WGE_gpu_img = cuda_call_WGE_kernel(src_grey, 2, 0.99, 32);

    if(save_imgs){
        imwrite(output_path + "/WGE_cpu_img_" + image_name, WGE_cpu_img); 
        imwrite(output_path + "/WGE_gpu_img_" + image_name, WGE_gpu_img);
    }

    for(unsigned int i = 0; i < ncols; i++){
        for(unsigned int j = 0; j < nrows; j++){
            assert(WGE_cpu_img.at<uchar>(i, j) == WGE_gpu_img.at<uchar>(i, j));
        }
    }

    cout << "Weak Gradient Elimination correctness test pass" << endl;
}

void test_WGE_speed(cv::Mat src_grey){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Measure CPU execution time
    auto cpu_start = high_resolution_clock::now();
    WGE(src_grey, 2, 0.99);
    auto cpu_end = high_resolution_clock::now();

    duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU Weak Gradient Elimination time: " << cpu_time.count() << " ms" << endl;

    // Measure GPU execution time
    auto gpu_start = high_resolution_clock::now();
    cuda_call_WGE_kernel(src_grey, 2, 0.99, 32);
    auto gpu_end = high_resolution_clock::now();

    duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    std::cout << "GPU Weak Gradient Elimination time: " << gpu_time.count() << " ms" << endl;

    std::cout << "Weak Gradient Elimination GPU speedup: " << cpu_time.count()/gpu_time.count() << "x" << endl;
}



void test_ST_correctness(cv::Mat src_grey, string image_name, string output_path, bool save_imgs){
    std::vector<cv::Point2f> ST_points_cpu = ST_corner_detection(src_grey, 30);
    std::vector<cv::Point2f> ST_points_gpu = cuda_call_ST_kernel(src_grey, 
                                                                 10, 
                                                                 20,
                                                                 10, 
                                                                 2,
                                                                 32);
    if(save_imgs){
        cv::Mat ST_img_cpu = src_grey.clone();
        cv::Mat ST_img_gpu = src_grey.clone();

        for( size_t i = 0; i < ST_points_cpu.size(); i++ )
        {
            circle(ST_img_cpu, ST_points_cpu[i], 10, Scalar(255), FILLED);
        }

        for( size_t i = 0; i < ST_points_gpu.size(); i++ )
        {
            circle(ST_img_gpu, ST_points_gpu[i], 10, Scalar(255), FILLED);
        }

        imwrite(output_path + "/ST_cpu_img_" + image_name, ST_img_cpu); 
        imwrite(output_path + "/ST_gpu_img_" + image_name, ST_img_gpu);
    }
    
    cout << "Verify Shi-Tomasi corectness with outputs" << endl;
}

void test_ST_speed(cv::Mat src_grey){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Measure CPU execution time
    auto cpu_start = high_resolution_clock::now();
    ST_corner_detection(src_grey, 30);
    auto cpu_end = high_resolution_clock::now();

    duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU Shi-Tomasi time: " << cpu_time.count() << " ms" << endl;;

    // Measure GPU execution time
    auto gpu_start = high_resolution_clock::now();
    cuda_call_ST_kernel(src_grey, 10000, 20, 10, 1, 32);
    auto gpu_end = high_resolution_clock::now();

    duration<double, std::milli> gpu_time = gpu_end - gpu_start;

    std::cout << "GPU Shi-Tomasi time: " << gpu_time.count() << " ms" << endl;

    std::cout << "Shi-Tomasi GPU speedup: " << cpu_time.count()/gpu_time.count() << "x" << endl;
}

int main(int argc, char* argv[]) {
    // Load in image
    string image_name = "satellite2.png";
    string input_path = "images/input/";
    string output_path = "images/output";

    // Read in image
    string image_path = input_path+image_name;
    cv::Mat src_grey = cv::imread(image_path, IMREAD_GRAYSCALE);
    cv::resize(src_grey, src_grey, cvSize(2048, 2048));

    // Test gradient operator
    cout << endl;
    test_gradient_operator_correctness(src_grey, image_name, output_path, true);
    test_gradient_operator_speed(src_grey);
    

    // Test histogram
    cout << endl;
    test_histogram_calculation_correctness(src_grey);
    test_histogram_calculation_speed(src_grey);

    // Test WGE
    cout << endl;
    test_WGE_correctness(src_grey, image_name, output_path, true);
    test_WGE_speed(src_grey);

    // Test Shi-Tomasi Corner Detection
    cout << endl;
    test_ST_correctness(src_grey, image_name, output_path, true);
    test_ST_speed(src_grey);

    cout << endl;
}
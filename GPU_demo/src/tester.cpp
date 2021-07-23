// #include "tester.hpp"

// void test_gradient_operator_correctness(cv::Mat src_grey, string image_name, string output_path){
//     cv::Mat gradient_cpu_img = prewittOperator(src_grey, 2);
//     cv::imwrite(output_path + "/prewitt_cpu_" + image_name, gradient_cpu_img); 

//     cv::Mat gradient_gpu_img = cuda_call_gradient_kernel(src_grey, 2, COMPOSITE, 32);
//     cv::imwrite(output_path + "/prewitt_gpu_" + image_name, gradient_gpu_img); 
// }

// void test_gradient_operator_speed(cv::Mat src_grey){
//     using std::chrono::high_resolution_clock;
//     using std::chrono::duration_cast;
//     using std::chrono::duration;
//     using std::chrono::milliseconds;

//     // Measure CPU execution time
//     auto cpu_start = high_resolution_clock::now();
//     cv::Mat gradient_cpu_img = prewittOperator(src_grey, 2);
//     auto cpu_end = high_resolution_clock::now();

//     duration<double, std::milli> cpu_time = cpu_end - cpu_start;

//     std::cout << "CPU gradient execution time: " << cpu_time.count() << " ms" << endl;;

//     // Measure GPU execution time
//     auto gpu_start = high_resolution_clock::now();
//     cv::Mat gradient_gpu_img = cuda_call_gradient_kernel(src_grey, 2, COMPOSITE, 32);
//     auto gpu_end = high_resolution_clock::now();

//     duration<double, std::milli> gpu_time = gpu_end - gpu_start;

//     std::cout << "GPU gradient execution time: " << gpu_time.count() << " ms" << endl;

//     std::cout << "Gradient operator GPU speedup: " << cpu_time.count()/gpu_time.count() << "x" << endl;
// }

// void test_histogram_calculation_speed(cv::Mat src_grey){
//     using std::chrono::high_resolution_clock;
//     using std::chrono::duration_cast;
//     using std::chrono::duration;
//     using std::chrono::milliseconds;

//     // Measure CPU execution time
//     auto cpu_start = high_resolution_clock::now();
//     unsigned int* hist_cpu = calculate_histogram(src_grey);
//     auto cpu_end = high_resolution_clock::now();

//     duration<double, std::milli> cpu_time = cpu_end - cpu_start;

//     std::cout << "CPU histogram calculation execution time: " << cpu_time.count() << " ms" << endl;;

//     // Measure GPU execution time
//     auto gpu_start = high_resolution_clock::now();
//     unsigned int* hist_gpu = cuda_call_hist_kernel(src_grey, 32);
//     auto gpu_end = high_resolution_clock::now();

//     duration<double, std::milli> gpu_time = gpu_end - gpu_start;

//     std::cout << "GPU histogram execution time: " << gpu_time.count() << " ms" << endl;
//     std::cout << "Histogram calculation GPU speedup: " << cpu_time.count()/gpu_time.count() << "x" << endl;
// }
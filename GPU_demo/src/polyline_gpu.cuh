#ifndef POLYLINE_GPU_CUH
#define POLYLINE_GPU_CUH

#include "cuda_header.cuh"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>

enum GradientDirection { HORIZONTAL, VERTICAL, COMPOSITE };

typedef struct eigenvalue_data{
    float ev;
    int i;
    int j;
} eigenvalue_data_t;

/**
 * GPU implementation of gradient operator.
 *
 * @param src_img a source image
 * @param proximity_term the weight given to closer pixels
 * @param thread_dim the number of threads per block dimension.
 * 
 * @return a new Mat object with the gradient values per pixel
 * 
 */
 cv::Mat cuda_call_gradient_kernel(cv::Mat src_img,
                                   int proximity_term,
                                   GradientDirection mode,
                                   unsigned int thread_dim);


/**
 * GPU implementation of gradient operator.
 *
 * @param src_img a source image
 * @param thread_dim the number of threads per block dimension.
 * 
 * @return an array of size 256 that holds the pixel 
 *         counts per intensity value
 * 
 */
unsigned int* cuda_call_hist_kernel(cv::Mat src_img, 
                                    unsigned int thread_dim);

/**
 * GPU implementation of weak gradient elimination.
 *
 * @param src_img a source image
 * @param grad_img image of the same dimensions with the gradients,
 * @param hist a histogram of pixel intensities 
 * @param percentile the minimum pixel percentile kept, and
 * @param thread_dim the number of threads per block dimension.
 * 
 * @return a new Mat object with the weak gradient pixels eliminated
 * 
 */
 cv::Mat cuda_call_WGE_kernel(cv::Mat src_img,
    int proximity_term,
    float percentile,
    unsigned int thread_dim);

/**
 * GPU implementation of Shi-Tomasi Corner Detection Algorithm.
 * Uses minimum of two eigenvalues for each Z matrix in order
 * to determine if a corner is valid. Assumes that the kernel
 * size is 3
 *
 * @param src_img A source image
 * @param grad_img An image with same dimensions as src_img with pixel gradients.
                   If null, then gradients will be calculated with proximity_term
 * @param threshold The threshold that candidate pixels must score over
 * @param proximity_term Non-negative weight given to adjacent pixels in gradient kernel. 
                         Only used if grad_img is NULL. If grad_img is NULL and 
                         proximity_term is NULL, then 1 will be used (Prewitt operator)
 * @param thread_dim the number of threads per block dimension.
 * 
 * @return an array of size 256 that holds the pixel 
 *         counts per intensity value
 * 
 */
std::vector<cv::Point2f> cuda_call_ST_kernel(cv::Mat src_img,
                                             float eig_threshold,
                                             float dist_threshold,
                                             unsigned int max_features,
                                             int proximity_term,
                                             unsigned int thread_dim);

#endif
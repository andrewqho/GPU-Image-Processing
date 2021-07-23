/*`
 line_detection.cpp
 polygon_detection
 
 Created by Andrew Ho on 6/24/18.
 Copyright © 2018 Andrew Ho, California Institute of Technology. All rights reserved.
 */

#include "polyline_cpu.hpp"

unsigned int* calculate_histogram(cv::Mat src_img){
    const unsigned int nrows = src_img.rows;
    const unsigned int ncols = src_img.cols;

    unsigned int* hist = (unsigned int*) malloc(256*sizeof(unsigned int));
    memset(hist, 0, 256*sizeof(unsigned int));

    for(unsigned int i = 0; i < ncols; i++){
        for(unsigned int j = 0; j < nrows; j++){
            unsigned int pix_val = src_img.at<uchar>(i, j);
            hist[pix_val] += 1;
        }
    }

    return hist;
}

/**
 Calculates histogram of input image and removes pixels below the specified threshold
 
 @param src_img : Source image
 @param threshold : Pixel value threshold
 
 @return removedGradient : Image after weak gradient pixels are removed
 */
cv::Mat WGE(cv::Mat src_img, int proximity_term, float percentile)
{   
    const unsigned int nrows = src_img.rows;
    const unsigned int ncols = src_img.cols;

    // Calculate grad_img
    cv::Mat grad_img = gradient_operator(src_img, proximity_term, 2);
    
    // calculate histogram for every pixel value (i.e [0 - 255])
    unsigned int* hist = calculate_histogram(grad_img);
    
    // Calculate threshold from histogram
    int threshold = 0;
    int pixel_count = 0;
    int pixel_threshold = percentile*nrows*ncols;

    while(pixel_count < pixel_threshold){
        pixel_count += hist[threshold];
        threshold += 1;
    }   
    
    cv::Mat removed_gradient = src_img.clone();

    for (int i = 0; i < removed_gradient.rows; i++)
    {
        for (int j = 0; j < removed_gradient.cols; j++)
        {
            int pixel = grad_img.at<uchar>(i,j);
            
            if(pixel < threshold)
            {
                removed_gradient.at<uchar>(i,j) = 0;
            }
        }
    }

    return removed_gradient;
}


cv::Mat gradient_operator(cv::Mat src_img, int proximity_term, int mode)
{   
    const unsigned int nrows = src_img.rows;
    const unsigned int ncols = src_img.cols;
    cv::Mat grad_img = cv::Mat::zeros(nrows, ncols, CV_8UC1);
    
    float dx;
    float dy; 
    int gradient;

    for(unsigned int i = 1; i < ncols-1; i++){
        for(unsigned int j = 1; j < nrows-1; j++){
            dx = 0;
            if(mode == 0 || mode == 2){
                dx = src_img.data[(j-1)*ncols + (i-1)] + proximity_term*src_img.data[j*ncols+(i-1)] + src_img.data[(j+1)*ncols+(i-1)] -
                     src_img.data[(j-1)*ncols + (i+1)] - proximity_term*src_img.data[j*ncols+(i+1)] - src_img.data[(j+1)*ncols+(i+1)];
            }

            dy = 0;
            if(mode == 1 || mode == 2){
                dy = src_img.data[(j-1)*ncols + (i-1)] + proximity_term*src_img.data[(j-1)*ncols+i] + src_img.data[(j-1)*ncols+(i+1)] -
                     src_img.data[(j+1)*ncols + (i-1)] - proximity_term*src_img.data[(j+1)*ncols+i] - src_img.data[(j+1)*ncols+(i+1)];
            }
            gradient = sqrt(dx*dx+dy*dy);

            if(gradient > 255){
                gradient = 255;
            }
            
            grad_img.at<uchar>(j, i) = gradient;
        }
    }
    return grad_img;
}


std::vector<cv::Point2f> ST_corner_detection(cv::Mat input, int maxCorners)
{
    /// Parameters for Shi-Tomasi algorithm
    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 21;
    bool useHarrisDetector = true;
    double k = 0.04;
    
    /// Apply corner detection
    goodFeaturesToTrack(input,
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        Mat(),
                        blockSize,
                        useHarrisDetector,
                        k );
    
    return corners;
}

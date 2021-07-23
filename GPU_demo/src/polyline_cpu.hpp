/*
 line_detection.hpp
 polygon_detection
 
 Created by Andrew Ho on 6/24/18.
 Copyright © 2018 Andrew Ho. All rights reserved.
 */

#ifndef line_detection_hpp
#define line_detection_hpp

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

/**
 Performs Shi-Tomasi Corner detection, detects a specified amount of corners
 
 @param maxCorners : Number of corners to detect
 @param input : Input image to find corners in
 
 @return corners : Vector of corner coordinates
 */
vector<Point2f> ST_corner_detection(Mat input, int maxCorners);

/**
 Helper function to apply convolution filter with specified coefficient
 
 @param src_img : Image to apply filter to
 @param proximity_term : Weight given to closer pixels
 @param mode : 0 for vertical, 1 for horizontal, 2 for composite
 
 @return Final product of convolutional filters
 */
Mat gradient_operator(Mat src_img, int proximity_term, int mode);

/**
 Removes pixels from source image below a certain gradient threshold
 
 @param src_img : Number of corners to detect
 @param proximity_term : proximity term for gradient calculation
 @param percentile : pixel percentile to remove below
 
 @return Copy of src_img with gradients removed
 */
Mat WGE(cv::Mat src_img, int proximity_term, float percentile);

/**
 * Calculates histogram of the input greyscale image
 * 
 * @param src_img : Input image to find corners in
 * 
 * @return static array of size 256 with pixel counts
 */
unsigned int* calculate_histogram(cv::Mat src_img);

#endif /* line_detection_hpp */


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
* Wrapper to call openCV shi-tomasi corner detection
*/
vector<Point2f> goodFeaturesToTrack_Callback(int maxCorners, Mat input );
Mat prewittOperator(Mat input_image, float coeff);
Mat WGE(Mat input_img, float threshold);

#endif /* line_detection_hpp */


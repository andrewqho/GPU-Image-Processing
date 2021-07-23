/*`
 line_detection.cpp
 polygon_detection
 
 Created by Andrew Ho on 6/24/18.
 Copyright © 2018 Andrew Ho, California Institute of Technology. All rights reserved.
 */

#include "line_detection.hpp"

/**
 Calculates histogram of input image and removes pixels below the specified threshold
 
 @param input_img : Source image
 @param threshold : Pixel value threshold
 
 @return removedGradient : Image after weak gradient pixels are removed
 */
Mat WGE(Mat input_img, float threshold)
{
    // calculate histogram for every pixel value (i.e [0 - 255])
    cv::Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::calcHist( &input_img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    
    // total pixels in image
    float totalPixels = input_img.cols * input_img.rows;
    
    // calculate percentage of every histogram bin (i.e: pixel value [0 - 255])
    // the 'bins' variable holds pairs of (int pixelV alue, float percentage)
    std::vector<std::pair<int, float>> bins;
    float percentage;
    for(int i = 0; i < 256; ++i)
    {
        percentage = (hist.at<float>(i,0)*100.0)/totalPixels;
        bins.push_back(std::make_pair(i, percentage));
    }
    
    // sort the bins according to percentage
    sort(bins.begin(), bins.end());
    
    // compute percentile for a pixel value
    
    Mat removedGradient = input_img.clone();
    
    for (int i = 0; i < input_img.rows; i++)
    {
        for (int j = 0; j < input_img.cols; j++)
        {
            int pixel = input_img.at<uchar>(i,j);
            
            float sum = 0;
            for (auto b : bins){
                if(b.first != pixel)
                {
                    sum += b.second;
                }
                else
                {
                    sum += b.second/2;
                    break;
                }
            }
            
            if(sum < threshold)
            {
                removedGradient.at<uchar>(i,j) = 0;
            }
        }
    }
    
    return removedGradient;
}

/**
 Helper function to apply convolution filter with specified coefficient
 
 @param input_image : Image to apply filter to
 @param coeff : Convlutional coefficient
 
 @return finalPrewitt : Final product of convolutional filters
 */
Mat prewittOperator(Mat input_image, float coeff)
{   
    Mat verticalPrewitt = input_image.clone();
    for (int i = 0; i < input_image.rows; i++)
    {
        for (int j = 0; j < input_image.cols; j++)
        {
            if(i == 0 || i == input_image.rows || j == 0 || j == input_image.cols)
            {
                verticalPrewitt.at<Vec3b>(i,j)[0] = 0;
                verticalPrewitt.at<Vec3b>(i,j)[1] = 0;
                verticalPrewitt.at<Vec3b>(i,j)[2] = 0;
            }
            else
            {
                int pixel1 = input_image.at<Vec3b>(i-1,j-1)[0] * -1;
                int pixel2 = input_image.at<Vec3b>(i,j-1)[0] * 0;
                int pixel3 = input_image.at<Vec3b>(i+1,j-1)[0] * 1;
                
                int pixel4 = input_image.at<Vec3b>(i-1,j)[0] * -1 * coeff;
                int pixel5 = input_image.at<Vec3b>(i,j)[0] * 0;
                int pixel6 = input_image.at<Vec3b>(i+1,j)[0] * 1 * coeff;
                
                int pixel7 = input_image.at<Vec3b>(i-1,j+1)[0] * -1;
                int pixel8 = input_image.at<Vec3b>(i,j+1)[0] * 0;
                int pixel9 = input_image.at<Vec3b>(i+1,j+1)[0] * 1;
                
                int sum = abs(pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9);
                sum = min(sum, 255);
                
                verticalPrewitt.at<Vec3b>(i,j)[0] = sum;
                verticalPrewitt.at<Vec3b>(i,j)[1] = sum;
                verticalPrewitt.at<Vec3b>(i,j)[2] = sum;
            }
        }
    }
    
    Mat horizontalPrewitt = input_image.clone();
    for (int i = 0; i < input_image.rows; i++)
    {
        for (int j = 0; j < input_image.cols; j++)
        {
            if(i == 0 || i == input_image.rows || j == 0 || j == input_image.cols)
            {
                horizontalPrewitt.at<Vec3b>(i, j)[0] = 0;
                horizontalPrewitt.at<Vec3b>(i, j)[1] = 0;
                horizontalPrewitt.at<Vec3b>(i, j)[2] = 0;
            }
            else
            {
                int pixel1 = input_image.at<Vec3b>(i-1,j-1)[0] * -1;
                int pixel2 = input_image.at<Vec3b>(i,j-1)[0] * -1 * coeff;
                int pixel3 = input_image.at<Vec3b>(i+1,j-1)[0] * -1;
            
                int pixel4 = input_image.at<Vec3b>(i-1,j)[0] * 0;
                int pixel5 = input_image.at<Vec3b>(i,j)[0] * 0;
                int pixel6 = input_image.at<Vec3b>(i+1,j)[0] * 0;
            
                int pixel7 = input_image.at<Vec3b>(i-1,j+1)[0] * 1;
                int pixel8 = input_image.at<Vec3b>(i,j+1)[0] * 1 * coeff;
                int pixel9 = input_image.at<Vec3b>(i+1,j+1)[0] * 1;
            
                int sum = abs(pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9);
                sum = min(sum, 255);
            
                horizontalPrewitt.at<Vec3b>(i,j)[0] = sum;
                horizontalPrewitt.at<Vec3b>(i,j)[1] = sum;
                horizontalPrewitt.at<Vec3b>(i,j)[2] = sum;
            }
        }
    }
    
    Mat finalPrewitt = verticalPrewitt + horizontalPrewitt;
    
    return finalPrewitt;
}

/**
 Performs Shi-Tomasi Corner detection, detects a specified amount of corners
 
 @param maxCorners : Number of corners to detect
 @param input : Input image to find corners in
 
 @return corners : Vector of corner coordinates
 */
vector<Point2f> goodFeaturesToTrack_Callback(int maxCorners, Mat input )
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

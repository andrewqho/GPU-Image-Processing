#include "line_detection.hpp"

/*
 Created by Andrew Ho on May 24, 2021.

 The below demo is a modified version of the full code that only demonstrates
 how to apply the prewitt operator, weak gradient elimination, and Shi-Tomasi corner
 detection. The full code is given at

https://github.com/andrewqho/PolylineDetection
 
 Copyright © 2018 Andrew Ho, California Institute of Technology. All rights reserved.
 */

int main(int argc, char* argv[]) {
    string image_name = "satellite1.png";
    string input_path = "images/input/";
    string output_path = "images/output";

    string image_path = input_path+image_name;
    Mat src = imread(image_path, IMREAD_COLOR);
    
    Mat src_grey;
    cvtColor(src, src_grey, COLOR_BGR2GRAY);
    cout << "Writing to " << output_path + "/grey_" + image_name << endl;
    imwrite(output_path + "/grey_" + image_name, src_grey); 

    // Apply prewitt operator
    Mat prewitt_img = prewittOperator(src, 1.0);
    imwrite(output_path + "/prewitt_" + image_name, prewitt_img); 

    // // Apply weak gradient elimination
    cvtColor(prewitt_img, prewitt_img, COLOR_BGR2GRAY);
    Mat wge_img = WGE(prewitt_img, 0.99);
    imwrite(output_path + "/WGE_" + image_name, wge_img);

    // Apply Shi-Tomasi Corner detection
    vector<Point2f> ST_points = goodFeaturesToTrack_Callback(30, wge_img);

    Mat ST_img = src.clone();
    for( size_t i = 0; i < ST_points.size(); i++ )
    {
        circle(ST_img, ST_points[i], 10, Scalar(0, 0, 255), FILLED );
    }
    imwrite(output_path + "/ST_" + image_name, ST_img);
        
}
//
//  FaceClassifier.cpp
//  FaceClassifier
//
//  Created by Ravindra Bhanot on 7/2/14.
//  Copyright (c) 2014 Rob-B. All rights reserved.
//

/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <dirent.h>
#include <vector>

using namespace cv;
using namespace std;

/** Global variables */
// Change the below string to the path of haar cascade xml found in the opencv library
string face_cascade_name = "<Path to haarcascade_frontalface_alt.xml>";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

vector<string> listdir (const char *path)
{
    // first off, we need to create a pointer to a directory
    DIR *pdir = NULL;
    pdir = opendir (path); // "." will refer to the current directory
    struct dirent *pent = NULL;
    vector<string> files;
    if (pdir == NULL) // if pdir wasn't initialised correctly
    { // print an error message and exit the program
        cout << "\nERROR! pdir could not be initialised correctly" << endl;
        return files; // exit the function
    } // end if
    
    while ((pent = readdir (pdir)) != NULL) // while there is still something in the directory to list
    {
        if (pent == NULL) // if pent has not been initialised correctly
        { // print an error message, and exit the program
            cout << "\nERROR! pent could not be initialised correctly" << endl;
            return files; // exit the function
        }
        
        // otherwise, it was initialised correctly. let's print it on the console:
        //cout << pent->d_name << "\n" << endl;
        //        if (pent->d_name.strcmp(".") == 0 || pent->d_name.strcmp("..") == 0)
        //            continue;
        /* On linux/Unix we don't want current and parent directories
         * If you're on Windows machine remove these lines
         */
        if (!strcmp (pent->d_name, "."))
            continue;
        if (!strcmp (pent->d_name, ".."))
            continue;
        if (!strcmp (pent->d_name, ".DS_Store"))
            continue;
        string filename = path;
        filename.append("/");
        filename.append(pent->d_name);
        files.push_back(filename);
    }
    
    // finally, let's close the directory
    closedir (pdir);
    return files;
}

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal-minVal+1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if(normed) {
        result /= (int)src.total();
    }
    return result.reshape(1,1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
    Mat src = _src.getMat();
    switch (src.type())
    {
        case CV_8SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_8UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_16SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_16UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_32SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_32FC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        default:
            CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;
    }
    return Mat();
}

static Mat spatial_histogram(InputArray _src, int numPatterns,                                                           int grid_x, int grid_y, bool /*normed*/)
{
    Mat src = _src.getMat();
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    if(src.empty())
        return result.reshape(1,1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
            Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
            // copy to the result matrix
            Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1,1);
}

//------------------------------------------------------------------------------
// LBPH
//------------------------------------------------------------------------------

template <typename _Tp> static
void olbp_(InputArray _src, OutputArray _dst) {
    // get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2, src.cols-2, CV_8UC1);
    Mat dst = _dst.getMat();
    // zero the result matrix
    dst.setTo(0);
    // calculate patterns
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
            code |= (src.at<_Tp>(i-1,j) >= center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
            code |= (src.at<_Tp>(i,j+1) >= center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
            code |= (src.at<_Tp>(i+1,j) >= center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
            code |= (src.at<_Tp>(i,j-1) >= center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
    //get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
    int type = src.type();
    switch (type) {
        case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
        case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
        case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
        case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
        case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
        case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
        case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
        default:
            string error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single channel images (given %d). Please pass the image data as a grayscale image!", type);
            CV_Error(CV_StsNotImplemented, error_msg);
            break;
    }
}

//------------------------------------------------------------------------------
// wrapper to cv::elbp (extended local binary patterns)
//------------------------------------------------------------------------------

static Mat elbp(InputArray src, int radius, int neighbors) {
    Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}


static Mat cropFaceFromImg(InputArray _frame)
{
    Mat testFrame = _frame.getMat();
    std::vector<Rect> faces;
    Mat frame_gray;
    std::vector<Mat> croppedFaces;
    Mat resizedFace;//dst image
    
    cvtColor( testFrame, frame_gray, CV_BGR2GRAY );
    //equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(1, 1) );
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceImg;
        Rect rect = Rect( faces[i].x + (faces[i].width/6), faces[i].y , faces[i].width*2/3, faces[i].height ); // ROI rect in srcImg
        frame_gray(rect).copyTo(faceImg);
        Size size(100,100);//the dst image size,e.g.100x100
        
        resize(faceImg,resizedFace,size);//resize image
        croppedFaces.push_back(resizedFace);
        
    }
    //imshow("ResizedFace", resizedFace);
    //waitKey(0);
    return resizedFace;
}

// Left for future use
/** @function thresh_callback */
void thresh_callback(int, void*, Mat src_gray )
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    //    /// Draw contours
    //    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    //    for( int i = 0; i< contours.size(); i++ )
    //    {
    //        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    //        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    //    }
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    for( int i = 0; i < contours.size(); i++ )
    { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    
    
    /// Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        //        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
        
    }
    
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    //-- Show what you got
    imshow( "Contours", drawing );
    // Wait for the user to press a key in the GUI window.
    cvWaitKey(0);
    // Free the resources.
    cvDestroyWindow("Image:");
    // cvReleaseImage(&input_Mat)
}



int main(int argc, const char *argv[]) {
    Mat input_img, input_face, input_query_face, input_lbp_image;
    Mat compare_img, compare_face_img, compare_lbp_image, compare_query_face;
    //Input image
    vector<string> photos = listdir ("<path to image direcotry which contains photos to be searched through>");
    string input = "<Path of input image containing the single face to be used for search>";
    
    // Process input image
    input_img = imread(input);
    /// Show in a window
    namedWindow( "Compare image", CV_WINDOW_AUTOSIZE );
    //-- Show what you got
    imshow( "Contours", input_img );
    // Wait for the user to press a key in the GUI window.
    cvWaitKey(0);
    // Free the resources.
    cvDestroyWindow("Image:");
    // cvReleaseImage(&input_Mat)
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    input_face= cropFaceFromImg(input_img );
    input_lbp_image = elbp(input_face, 1, 8);
    input_query_face = spatial_histogram(input_lbp_image, /* lbp_image */
                                         static_cast<int>(std::pow(2.0, static_cast<double>(8))), /* number of possible patterns */
                                         8, /* grid size x */
                                         8, /* grid size y */
                                         true /* normed histograms */);
    
    for(vector<string>::const_iterator i = photos.begin(); i != photos.end(); ++i) {
        cout << *i << endl;
        compare_img = imread(*i);
        compare_face_img = cropFaceFromImg(compare_img);
        
        //thresh_callback( 0, 0, face_img_2 );
        // get the spatial histogram from input image
        compare_lbp_image = elbp(compare_face_img, 1, 8);
        compare_query_face = spatial_histogram(compare_lbp_image, /* lbp_image */
                                               static_cast<int>(std::pow(2.0, static_cast<double>(8))), /* number of possible patterns */
                                               8, /* grid size x */
                                               8, /* grid size y */
                                               true /* normed histograms */);
        double dist = compareHist(input_query_face, compare_query_face, CV_COMP_CORREL );
        // cout for statistical purposes
        cout << dist;
        if (dist > 0.6) {
            /// Show in a window
            namedWindow( "Compare image", CV_WINDOW_AUTOSIZE );
            //-- Show what you got
            imshow( "Contours", compare_img );
            // Wait for the user to press a key in the GUI window.
            cvWaitKey(0);
            // Free the resources.
            cvDestroyWindow("Image:");
            // cvReleaseImage(&input_Mat)
        }
    }
}
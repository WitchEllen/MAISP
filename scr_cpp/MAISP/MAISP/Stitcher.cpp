#include <stdio.h>  
#include <iostream>  
#include <vector>  

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/stitching.hpp" 
using namespace cv;
using namespace std;

int main() {

    Mat img_1 = imread("1.jpg");
    Mat img_2 = imread("2.jpg");

    vector<Mat> imgs;

    imgs.push_back(img_1);
    imgs.push_back(img_2);

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    stitcher->estimateTransform(imgs);
    //stitcher.composePanorama(pano); //use old images
    stitcher->composePanorama(imgs, pano); //use new images with same size and count
    imwrite("pano.jpg", pano);
    return 0;
}
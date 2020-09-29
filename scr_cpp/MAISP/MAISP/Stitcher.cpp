#include <vector>  
#include "mStitcher.h"  
#include "opencv2/stitching.hpp"

using namespace cv;
using namespace std;

int main() {

    Mat img_1 = imread("D:/Program/GitHub/MAISP/data/left.jpg");
    Mat img_2 = imread("D:/Program/GitHub/MAISP/data/right.jpg");
    Mat img_3 = imread("D:/Program/GitHub/MAISP/data/left2.jpg");
    Mat img_4 = imread("D:/Program/GitHub/MAISP/data/right2.jpg");

    vector<Mat> imgs_init;
    vector<Mat> imgs;

    imgs_init.push_back(img_1);
    imgs_init.push_back(img_2);
    imgs.push_back(img_3);
    imgs.push_back(img_4);

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    stitcher->estimateTransform(imgs);
    stitcher->composePanorama(pano); //use old images
    //stitcher->composePanorama(imgs, pano); //use new images with same size and count
    imwrite("D:/Program/GitHub/MAISP/result/pano.jpg", pano);
    mStitch(img_4, img_3, "D:/Program/GitHub/MAISP/result/");
    return 0;
}


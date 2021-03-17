#pragma once
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, four_corners_t& corners);

void CalcCorners(const Mat& H, const Mat& src, four_corners_t& corners)
{
    double v2[] = { 0, 0, 1 };  //���Ͻ�
    double v1[3];   //�任�������ֵ
    Mat V2 = Mat(3, 1, CV_64FC1, v2);   //������
    Mat V1 = Mat(3, 1, CV_64FC1, v1);   //������

    //���Ͻ�(0,0,1)
    V1 = H * V2;
    cout << "V2: " << V2 << endl;
    cout << "V1: " << V1 << endl;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    //���½�(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);   //������
    V1 = Mat(3, 1, CV_64FC1, v1);   //������
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    //���Ͻ�(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);   //������
    V1 = Mat(3, 1, CV_64FC1, v1);   //������
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    //���½�(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);   //������
    V1 = Mat(3, 1, CV_64FC1, v1);   //������
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];
}

void mStitch(Mat& image01, Mat& image02, String dst_dir)
{
    //�Ҷ�ͼת��
    Mat image1, image2;
    cvtColor(image01, image1, COLOR_BGR2GRAY);
    cvtColor(image02, image2, COLOR_BGR2GRAY);

    //��ȡ�����㲢��������   
    Ptr<SURF> Detector = SURF::create(2000);
    vector<KeyPoint> keyPoint1, keyPoint2;
    Mat imageDesc1, imageDesc2;
    Detector->detectAndCompute(image1, Mat(), keyPoint1, imageDesc1);
    Detector->detectAndCompute(image2, Mat(), keyPoint2, imageDesc2);

    FlannBasedMatcher matcher1, matcher2;
    vector<vector<DMatch> > matchePoints;
    vector<DMatch> GoodMatchePoints;
    vector<Mat> train_desc1(1, imageDesc1);
    vector<Mat> train_desc2(1, imageDesc2);
    matcher1.add(train_desc1);
    matcher2.add(train_desc2);
    set<pair<int,int> > matches;

    matcher1.train();
    matcher1.knnMatch(imageDesc2, matchePoints, 2);
    // Lowe's algorithm,��ȡ����ƥ���
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
        {
            GoodMatchePoints.push_back(matchePoints[i][0]);
            matches.insert(make_pair(matchePoints[i][0].queryIdx, matchePoints[i][0].trainIdx));
        }
    }
    cout << endl << "1->2 matches: " << GoodMatchePoints.size() << endl;

    matchePoints.clear();
    matcher2.train();
    matcher2.knnMatch(imageDesc1, matchePoints, 2);
    // Lowe's algorithm,��ȡ����ƥ���
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
        {
            if (matches.find(make_pair(matchePoints[i][0].trainIdx, matchePoints[i][0].queryIdx)) == matches.end())
            {
                GoodMatchePoints.push_back(DMatch(matchePoints[i][0].trainIdx, matchePoints[i][0].queryIdx, matchePoints[i][0].distance));
            }
        }
    }
    cout << "1->2 & 2->1 matches: " << GoodMatchePoints.size() << endl;

    Mat match_result;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, match_result);
    imwrite(dst_dir + "match_result.jpg", match_result);

    vector<Point2f> imagePoints1, imagePoints2;

    for (int i = 0; i < GoodMatchePoints.size(); i++)
    {
        imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
    }

    //��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
    Mat homo = findHomography(imagePoints1, imagePoints2, RANSAC);

    //������׼ͼ���ĸ���������
    four_corners_t corners;
    CalcCorners(homo, image01, corners);
    cout << "left_top:" << corners.left_top << endl;
    cout << "left_bottom:" << corners.left_bottom << endl;
    cout << "right_top:" << corners.right_top << endl;
    cout << "right_bottom:" << corners.right_bottom << endl;

    //ͼ����׼  
    Mat imageTransform1, imageTransform2;
    warpPerspective(image01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
    //imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);
    imwrite(dst_dir + "trans1.jpg", imageTransform1);

    //����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
    int dst_width = imageTransform1.cols;  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
    int dst_height = image02.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));
    imwrite(dst_dir + "b_dst.jpg", dst);

    OptimizeSeam(image02, imageTransform1, dst, corners);
    imwrite(dst_dir + "dst.jpg", dst);
}

//�Ż���ͼ�����Ӵ���ʹ��ƴ����Ȼ
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, four_corners_t& corners)
{
    int start = MIN(corners.left_top.x, corners.left_bottom.x);//��ʼλ�ã����ص��������߽�  
    double processWidth = img1.cols - start;//�ص�����Ŀ��  
    int rows = dst.rows;
    int cols = img1.cols; //ע�⣬������*ͨ����
    double alpha = 1;//img1�����ص�Ȩ��  
    for (int i = 0; i < rows; i++)
    {
        uchar* p = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
        uchar* t = trans.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = start; j < cols; j++)
        {
            //�������ͼ��trans�������صĺڵ㣬����ȫ����img1�е�����
            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
            {
                alpha = 1;
            }
            else
            {
                //img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ�������ȣ�ʵ��֤�������ַ���ȷʵ��  
                alpha = (processWidth - (j - start)) / processWidth;
            }
            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
        }
    }
}
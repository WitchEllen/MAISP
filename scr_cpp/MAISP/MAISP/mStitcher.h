#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
//#include "opencv2/legacy/legacy.hpp"

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::detail;
using namespace cv::xfeatures2d;

Mat stitch(vector<Mat> imgs)
{
    int num_images = imgs.size();

    // old version
    // Ptr<FeaturesFinder> finder;
    // finder = new SurfFeaturesFinder();

    Ptr<SURF> finder = SURF::create();  // SURF
    vector<ImageFeatures> features(num_images);
    for (int i = 0; i < num_images; i++) {
        // (*finder)(imgs[i], features[i]); // old version
        computeImageFeatures(finder, imgs[i], features[i]);
    }

    vector<MatchesInfo> pairwise_matches;               //����ƥ����Ϣ
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);   //��������ƥ������2NN����
    matcher(features, pairwise_matches);                //��������ƥ��

    HomographyBasedEstimator estimator;                 //�������������
    vector<CameraParams> cameras;                       //��ʾ�������
    estimator(features, pairwise_matches, cameras);     //���������������

    for (size_t i = 0; i < cameras.size(); ++i)         //ת�������ת��������������
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;           //����ƽ�����ȷ�������
    adjuster = new detail::BundleAdjusterReproj();      //��ӳ������
    //adjuster = new detail::BundleAdjusterRay();       //���߷�ɢ����

    adjuster->setConfThresh(1);                         //����ƥ�����Ŷȣ���ֵ��Ϊ1
    (*adjuster)(features, pairwise_matches, cameras);   //��ȷ�����������

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)         //�����������ת����
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);             //���в���У��
    for (size_t i = 0; i < cameras.size(); ++i)         //���������ֵ
        cameras[i].R = rmats[i];
    rmats.clear();

    vector<Point> corners(num_images);      //��ʾӳ��任��ͼ������Ͻ�����
    vector<UMat> masks_warped(num_images);  //��ʾӳ��任���ͼ������
    vector<UMat> images_warped(num_images); //��ʾӳ��任���ͼ��
    vector<Size> sizes(num_images);         //��ʾӳ��任���ͼ��ߴ�
    vector<UMat> masks(num_images);         //��ʾԴͼ������

    for (int i = 0; i < num_images; ++i)    //��ʼ��Դͼ������
    {
        masks[i].create(imgs[i].size(), CV_8U);     //����ߴ��С
        masks[i].setTo(Scalar::all(255));           //ȫ����ֵΪ255����ʾԴͼ����������ʹ��
    }

    Ptr<WarperCreator> warper_creator;                  //����ͼ��ӳ��任������
    //warper_creator = new cv::PlaneWarper();           //ƽ��ͶӰ
    //warper_creator = new cv::CylindricalWarper();     //����ͶӰ
    warper_creator = new cv::SphericalWarper();       //����ͶӰ
    //warper_creator = new cv::FisheyeWarper();         //����ͶӰ
    //warper_creator = new cv::StereographicWarper();   //������ͶӰ

    //����ͼ��ӳ��任��������ӳ��ĳ߶�Ϊ����Ľ��࣬��������Ľ��඼��ͬ
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);    //ת������ڲ�������������
        //�Ե�ǰͼ����ͶӰ�任���õ��任���ͼ���Լ���ͼ������Ͻ�����
        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();     //�õ��ߴ�
        //�õ��任���ͼ������
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    imgs.clear();
    masks.clear();

    // �����عⲹ������Ӧ�����油������
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
    compensator->feed(corners, images_warped, masks_warped);    //�õ��عⲹ����
    for (int i = 0; i < num_images; ++i)    //Ӧ���عⲹ��������ͼ������عⲹ��
    {
        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
    }

    //�ں��棬���ǻ���Ҫ�õ�ӳ��任ͼ������masks_warped���������Ϊ�ñ������һ������masks_seam
    vector<UMat> masks_seam(num_images);
    for (int i = 0; i < num_images; i++)
        masks_warped[i].copyTo(masks_seam[i]);

    Ptr<SeamFinder> seam_finder;                                //����ӷ���Ѱ����
   //seam_finder = new NoSeamFinder();                          //����Ѱ�ҽӷ���
   //seam_finder = new VoronoiSeamFinder();                     //��㷨
   //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);       //��̬�淶��
   //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
   //ͼ�
   //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
    seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)    //ͼ����������ת��
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    images_warped.clear();

    //�õ��ӷ��ߵ�����ͼ��masks_seam
    seam_finder->find(images_warped_f, corners, masks_seam);

    vector<Mat> images_warped_s(num_images);
    Ptr<Blender> blender;    //����ͼ���ں���

    blender = Blender::createDefault(Blender::NO, false);    //���ںϷ���
    //���ںϷ���
    //blender = Blender::createDefault(Blender::FEATHER, false);
    //FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
    //fb->setSharpness(0.005);    //���������

    //blender = Blender::createDefault(Blender::MULTI_BAND, false);    //��Ƶ���ں�
    //MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
    //mb->setNumBands(8);   //����Ƶ������������������

    blender->prepare(corners, sizes);    //����ȫ��ͼ������

    //���ںϵ�ʱ������Ҫ�����ڽӷ���������д�������һ����Ѱ�ҽӷ��ߺ�õ�������ı߽���ǽӷ��ߴ���������ǻ���Ҫ�ڽӷ������࿪��һ�����������ںϴ�����һ������̶��𻯷�����Ϊ�ؼ�
    //Ӧ�������㷨��С�������
    vector<Mat> dilate_img(num_images);
    Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //����ṹԪ��
    for (int k = 0; k < num_images; k++)
    {
        images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //�ı���������
        dilate(masks_seam[k], masks_seam[k], element);    //��������
        //ӳ��任ͼ����������ͺ�������ࡰ�롱���Ӷ�ʹ��չ������������ڽӷ������࣬�����߽紦����Ӱ��
        // masks_seam[k] = masks_seam[k] & masks_warped[k];
        //Mat mask_seam, mask_warped;
        //masks_seam[k].copyTo(mask_seam);
        //masks_warped[k].copyTo(mask_warped);
        //mask_seam = mask_seam & mask_warped;
        //mask_seam.copyTo(masks_seam[k]);
        blender->feed(images_warped_s[k], masks_seam[k], corners[k]);    //��ʼ������
    }

    masks_seam.clear();
    images_warped_s.clear();
    masks_warped.clear();
    images_warped_f.clear();

    Mat result, result_mask;
    //����ںϲ������õ�ȫ��ͼ��result����������result_mask
    blender->blend(result, result_mask);

    return result;
}
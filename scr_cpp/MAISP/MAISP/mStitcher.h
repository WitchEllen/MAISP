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

    vector<MatchesInfo> pairwise_matches;               //特征匹配信息
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);   //定义特征匹配器，2NN方法
    matcher(features, pairwise_matches);                //进行特征匹配

    HomographyBasedEstimator estimator;                 //定义参数评估器
    vector<CameraParams> cameras;                       //表示相机参数
    estimator(features, pairwise_matches, cameras);     //进行相机参数评估

    for (size_t i = 0; i < cameras.size(); ++i)         //转换相机旋转参数的数据类型
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;           //光束平差法，精确相机参数
    adjuster = new detail::BundleAdjusterReproj();      //重映射误差方法
    //adjuster = new detail::BundleAdjusterRay();       //射线发散误差方法

    adjuster->setConfThresh(1);                         //设置匹配置信度，该值设为1
    (*adjuster)(features, pairwise_matches, cameras);   //精确评估相机参数

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)         //复制相机的旋转参数
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);             //进行波形校正
    for (size_t i = 0; i < cameras.size(); ++i)         //相机参数赋值
        cameras[i].R = rmats[i];
    rmats.clear();

    vector<Point> corners(num_images);      //表示映射变换后图像的左上角坐标
    vector<UMat> masks_warped(num_images);  //表示映射变换后的图像掩码
    vector<UMat> images_warped(num_images); //表示映射变换后的图像
    vector<Size> sizes(num_images);         //表示映射变换后的图像尺寸
    vector<UMat> masks(num_images);         //表示源图的掩码

    for (int i = 0; i < num_images; ++i)    //初始化源图的掩码
    {
        masks[i].create(imgs[i].size(), CV_8U);     //定义尺寸大小
        masks[i].setTo(Scalar::all(255));           //全部赋值为255，表示源图的所有区域都使用
    }

    Ptr<WarperCreator> warper_creator;                  //定义图像映射变换创造器
    //warper_creator = new cv::PlaneWarper();           //平面投影
    //warper_creator = new cv::CylindricalWarper();     //柱面投影
    warper_creator = new cv::SphericalWarper();       //球面投影
    //warper_creator = new cv::FisheyeWarper();         //鱼眼投影
    //warper_creator = new cv::StereographicWarper();   //立方体投影

    //定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
        //对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();     //得到尺寸
        //得到变换后的图像掩码
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    imgs.clear();
    masks.clear();

    // 创建曝光补偿器，应用增益补偿方法
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
    compensator->feed(corners, images_warped, masks_warped);    //得到曝光补偿器
    for (int i = 0; i < num_images; ++i)    //应用曝光补偿器，对图像进行曝光补偿
    {
        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
    }

    //在后面，我们还需要用到映射变换图的掩码masks_warped，因此这里为该变量添加一个副本masks_seam
    vector<UMat> masks_seam(num_images);
    for (int i = 0; i < num_images; i++)
        masks_warped[i].copyTo(masks_seam[i]);

    Ptr<SeamFinder> seam_finder;                                //定义接缝线寻找器
   //seam_finder = new NoSeamFinder();                          //无需寻找接缝线
   //seam_finder = new VoronoiSeamFinder();                     //逐点法
   //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);       //动态规范法
   //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
   //图割法
   //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
    seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)    //图像数据类型转换
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    images_warped.clear();

    //得到接缝线的掩码图像masks_seam
    seam_finder->find(images_warped_f, corners, masks_seam);

    vector<Mat> images_warped_s(num_images);
    Ptr<Blender> blender;    //定义图像融合器

    blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
    //羽化融合方法
    //blender = Blender::createDefault(Blender::FEATHER, false);
    //FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
    //fb->setSharpness(0.005);    //设置羽化锐度

    //blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
    //MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
    //mb->setNumBands(8);   //设置频段数，即金字塔层数

    blender->prepare(corners, sizes);    //生成全景图像区域

    //在融合的时候，最重要的是在接缝线两侧进行处理，而上一步在寻找接缝线后得到的掩码的边界就是接缝线处，因此我们还需要在接缝线两侧开辟一块区域用于融合处理，这一处理过程对羽化方法尤为关键
    //应用膨胀算法缩小掩码面积
    vector<Mat> dilate_img(num_images);
    Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //定义结构元素
    for (int k = 0; k < num_images; k++)
    {
        images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //改变数据类型
        dilate(masks_seam[k], masks_seam[k], element);    //膨胀运算
        //映射变换图的掩码和膨胀后的掩码相“与”，从而使扩展的区域仅仅限于接缝线两侧，其他边界处不受影响
        // masks_seam[k] = masks_seam[k] & masks_warped[k];
        //Mat mask_seam, mask_warped;
        //masks_seam[k].copyTo(mask_seam);
        //masks_warped[k].copyTo(mask_warped);
        //mask_seam = mask_seam & mask_warped;
        //mask_seam.copyTo(masks_seam[k]);
        blender->feed(images_warped_s[k], masks_seam[k], corners[k]);    //初始化数据
    }

    masks_seam.clear();
    images_warped_s.clear();
    masks_warped.clear();
    images_warped_f.clear();

    Mat result, result_mask;
    //完成融合操作，得到全景图像result和它的掩码result_mask
    blender->blend(result, result_mask);

    return result;
}
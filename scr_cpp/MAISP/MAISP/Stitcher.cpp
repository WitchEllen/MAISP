#include <vector>
#include <fstream>
#include <string>
#include <regex>
#include <filesystem>
#include "mStitcher.h"  
#include "opencv2/stitching.hpp"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void getAllPaths(string path, vector<string>& files, string format)
{
    transform(format.begin(), format.end(), format.begin(), ::tolower);
    regex fileSuffix_l("(.*)(." + format + ")");
    transform(format.begin(), format.end(), format.begin(), ::toupper);
    regex fileSuffix_u("(.*)(." + format + ")");
    for (const auto& entry : fs::directory_iterator(path))
    {
        fs::path path = entry.path();
        if (std::regex_match(path.string(), fileSuffix_l) || std::regex_match(path.string(), fileSuffix_u))
        {
            files.push_back(path.string());
        }
    }
}

int main() {
    fstream setting;
    setting.open("setting.txt", ios::in);

    string input_dir, output_path, name;
    getline(setting, input_dir);
    getline(setting, output_path);
    getline(setting, name);
    input_dir = input_dir + name + "/";

    vector<string> files_path;
    getAllPaths(input_dir, files_path, ".jpg");

    vector<Mat> imgs_init;
    vector<Mat> imgs;

    for (auto file_path: files_path)
        imgs.push_back(imread(file_path));

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    //stitcher->stitch(imgs, pano);
    //stitcher->estimateTransform(imgs);
    //stitcher->composePanorama(pano); //use old images
    //imwrite(output_path + name + "_m.png", pano);
    //stitcher->setBlender(Blender::createDefault(Blender::FEATHER, true));
    //stitcher->stitch(imgs, pano);
    //imwrite(output_path + name + "_f.png", pano);
    //stitcher->setBlender(Blender::createDefault(Blender::NO, true));
    //stitcher->stitch(imgs, pano);
    //imwrite(output_path + name + ".png", pano);
    // stitcher->composePanorama(imgs, pano); //use new images with same size and count
    pano = stitch(imgs);
    imwrite(output_path + name + "_n_nomask.png", pano);
    return 0;
}

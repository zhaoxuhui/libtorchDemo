#include <iostream>

// torch相关引用
#include "torch/torch.h"
#include "torch/script.h"

// OpenCV相关引用
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace torch;
using namespace std;
using namespace cv;

int main() {
    // 感谢Wu Xin提供的训练好的DCE网络模型(打包Torch版本1.0.0)
    string model_path = "../DCE.pt";
    string img_path = "../test.png";

    // 加载模型
    cout << "Loading low-light model" << endl;
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);
    cout << "Initialized low-light model" << endl;

    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    Mat im = imread(img_path, IMREAD_COLOR);
    Mat normedImg;
    // 影像灰度归一化，img是OpenCV的float32的Mat类型
    im.convertTo(normedImg, CV_32FC3, 1.f / 255.f, 0);

    int img_width = im.cols;
    int img_height = im.rows;

    // 将OpenCV的Mat类型构造成Tensor，然后再转换成可求导的变量
    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(normedImg.data, {1, img_height, img_width, 3});
    img_tensor = img_tensor.permute({0, 3, 2, 1});
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(device);

    // 前向推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var);
    auto output = module->forward(inputs).toTuple();

    // 后处理
    auto img_enhanced = output->elements()[1].toTensor().to(torch::kCPU).squeeze();
    cv::Mat img_enhanced_cv(cv::Size(img_height, img_width), CV_32FC1, img_enhanced.data<float>());

    cv::Mat img_post;
    cv::transpose(img_enhanced_cv, img_post);

    cv::Mat final_img(img_width, img_height, CV_8UC3);
    img_post = img_post * 255;
    img_post.convertTo(final_img, CV_8UC3);
    final_img.convertTo(final_img, CV_8UC1);

    imshow("enhanced img", final_img);
    imshow("raw img", im);
    waitKey(0);

    return 0;
}

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "pose_estimation.hpp"
#include "render_human_pose.hpp"
#include<map>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace human_pose_estimation; 
using std::string;

void video_demo(PoseEstimator pose_estimator, map<string, string> params);
void image_demo(PoseEstimator pose_estimator, map<string, string> params);

int main(int argc, char** argv){
    map<string, string> params;
    if(argc!=4){
        params["trained_file"] = "/home/xyliu/cpp/caffe/projects/poseEstimation/tinypose.caffemodel";
        params["model_file"] = "/home/xyliu/cpp/caffe/projects/poseEstimation/tinypose.prototxt";
        params["img_test"] = "/home/xyliu/cvToolBox/data/test.png";
    }
    else{
        params["model_file"] = argv[1];
        params["trained_file"] = argv[2];
        params["img_test"] = argv[3];
    }

    PoseEstimator pose_estimator(params);
    image_demo(pose_estimator, params);
    video_demo(pose_estimator, params);
    return 0;
}


void image_demo(PoseEstimator pose_estimator, map<string, string> params){
    const string file = params["img_test"]; 
    cv::Mat img = cv::imread(file, -1);
    std::vector<HumanPose> poses = pose_estimator.poseEstimation(img);
    renderHumanPose(poses, img);
    int delay = 3333;
    cv::imshow("ICV Human Pose Estimation", img);
    // cv::imwrite("saved.jpg", img);
    cv::waitKey(delay);
}


void video_demo(PoseEstimator pose_estimator, map<string, string> params){
    try{
        cv::VideoCapture cap("/home/xyliu/cvToolBox/data/test.mp4");

        int delay = 33;
        double inferenceTime = 0.0;
        cv::Mat image;
        if (!cap.read(image)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }

        do {
            double t1 = static_cast<double>(cv::getTickCount());
            std::vector<HumanPose> poses = pose_estimator.poseEstimation(image);
            double t2 = static_cast<double>(cv::getTickCount());
            // std::cout << "infrence time is: " << (t2-t1) << std::endl;
            if (inferenceTime == 0) {
                inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
            } else {
                inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
            }

            renderHumanPose(poses, image);

            cv::Mat fpsPane(35, 155, CV_8UC3);
            fpsPane.setTo(cv::Scalar(153, 119, 76));
            cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
            cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
            std::stringstream fpsSs;
            fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
            cv::putText(image, fpsSs.str(), cv::Point(16, 32),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
            cv::imshow("ICV Human Pose Estimation", image);
            // cv::imwrite("saved.jpg", image);

            int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (key == 27) {
                break;
            }
        } while (cap.read(image));
    // std::cout << "infrence time is: " << inferenceTime << std::endl;
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
    }

}

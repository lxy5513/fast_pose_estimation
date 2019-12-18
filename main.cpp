#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "pose_estimation.hpp"
#include "render_human_pose.hpp"


using namespace caffe;  // NOLINT(build/namespaces)
using namespace human_pose_estimation; 
using std::string;

int main(int argc, char** argv){
    PoseEstimator pose_estimator;
    const string file = "/home/xyliu/cvToolBox/data/test.png";
    cv::Mat img = cv::imread(file, -1);
    std::vector<HumanPose> poses = pose_estimator.poseEstimation(img);
    renderHumanPose(poses, img);

    int delay = 3333;
    cv::imshow("ICV Human Pose Estimation", img);
    int key = cv::waitKey(delay) & 255;

    return 0;
}

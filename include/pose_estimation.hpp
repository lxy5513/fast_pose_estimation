#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <caffe/caffe.hpp>

#include "human_pose.hpp"


using namespace caffe;  // NOLINT(build/namespaces)
typedef std::vector<caffe::Blob<float>*> ModelOutput;

namespace human_pose_estimation {
class PoseEstimator{
    private:
        ModelOutput Predict(const cv::Mat& img);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
        void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
        void correctCoordinates(std::vector<HumanPose>& poses,
                                const cv::Size& featureMapsSize,
                                const cv::Size& imageSize) const;
        std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
        std::vector<HumanPose> Postprocess(ModelOutput model_result, const cv::Mat& img);



        shared_ptr<Net<float> > net_;
        int num_channels_;
        cv::Size input_geometry_;
        int minJointsNumber;
        int stride;
        cv::Vec4i pad;
        cv::Vec3f meanPixel;
        float minPeaksDistance;
        float midPointsScoreThreshold;
        float foundMidPointsRatioThreshold;
        float minSubsetScore;
        cv::Size inputLayerSize;
        int upsampleRatio;

        
    public:
        static const size_t keypointsNumber;
        PoseEstimator(map<string, string> params);
        std::vector<HumanPose> poseEstimation(const cv::Mat& img);
};

}//namespace human_pose_estimation


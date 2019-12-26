#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "peak.hpp"
#include "pose_estimation.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

namespace human_pose_estimation {

const size_t PoseEstimator::keypointsNumber = 18;

PoseEstimator::PoseEstimator(map<string, string> params)
    : minJointsNumber(3),
      stride(8),
      pad(cv::Vec4i::all(0)),
      meanPixel(cv::Vec3f::all(128)),
      minPeaksDistance(3.0f),
      midPointsScoreThreshold(0.05f),
      foundMidPointsRatioThreshold(0.8f),
      minSubsetScore(0.2f),
      inputLayerSize(-1, -1),
      upsampleRatio(4){

    Caffe::set_mode(Caffe::GPU);
    /* Load the network. */
    const string model_file = params["model_file"];
    const string trained_file = params["trained_file"];

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    std::cout << "width is " << input_layer->width() <<"\t height is " << input_layer->height() << std::endl;
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}


ModelOutput PoseEstimator::Predict(const cv::Mat& img){
    double t1 = static_cast<double>(cv::getTickCount());
    net_->Forward();
    double t2 = static_cast<double>(cv::getTickCount());
    double inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
    std::cout << "model infrence time is: " << inferenceTime << std::endl;

    Blob<float>* output_heatmaps_ = net_->output_blobs()[0]; //  display output_heatmaps.shape_  is  {1, 19, 32, 43}
    Blob<float>* output_pafs_ = net_->output_blobs()[1]; // shape is  {1, 38, 32, 43}
    
    ModelOutput result;
    result.push_back(output_heatmaps_);
    result.push_back(output_pafs_);
    // std::cout << "get heatmap and pafs" << std::endl;
    return result;
}


std::vector<HumanPose> PoseEstimator::Postprocess(ModelOutput model_result, const cv::Mat& img){
    Blob<float>* output_heatmaps = model_result[0];
    Blob<float>* output_pafs = model_result[1];
    const float* heatMapsData = output_heatmaps->cpu_data();
    const int heatMapOffset = output_heatmaps->height() * output_heatmaps->width();
    const int nHeatMaps = keypointsNumber;
    const float* pafsData = output_pafs->cpu_data();
    const int pafOffset = output_pafs->height() * output_pafs->width();
    const int nPafs = output_pafs->channels();
    const int featureMapWidth = output_heatmaps->width();
    const int featureMapHeight = output_heatmaps->height();
    const cv::Size& imageSize = img.size();

    std::vector<cv::Mat> heatMaps(nHeatMaps);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                              reinterpret_cast<void*>(
                                  const_cast<float*>(
                                      heatMapsData + i * heatMapOffset)));
    }
    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(nPafs);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                          reinterpret_cast<void*>(
                              const_cast<float*>(
                                  pafsData + i * pafOffset)));
    }
    resizeFeatureMaps(pafs);

    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);
    correctCoordinates(poses, heatMaps[0].size(), imageSize);
    return poses;
}

void PoseEstimator::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
    for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}


void PoseEstimator::correctCoordinates(std::vector<HumanPose>& poses,
                                            const cv::Size& featureMapsSize,
                                            const cv::Size& imageSize) const {
    CV_Assert(stride % upsampleRatio == 0);

    cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

    float scaleX = imageSize.width /
            static_cast<float>(fullFeatureMapSize.width - pad(1) - pad(3));
    float scaleY = imageSize.height /
            static_cast<float>(fullFeatureMapSize.height - pad(0) - pad(2));
    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= stride / upsampleRatio;
                keypoint.x -= pad(1);
                keypoint.x *= scaleX;

                keypoint.y *= stride / upsampleRatio;
                keypoint.y -= pad(0);
                keypoint.y *= scaleY;
            }
        }
    }
}


void PoseEstimator::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels){
    cv::Mat sample = img;
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_normalized;
    sample_resized.convertTo(sample_normalized, CV_32FC3, 1.0, -128);
    sample_normalized.convertTo(sample_normalized, CV_32FC3, 1/255.0, 0);


    // This operation will write the separate BGR planes directly to the
   // input layer of the network because it is wrapped by the cv::Mat
   // objects in input_channels. 
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
            == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void PoseEstimator::WrapInputLayer(std::vector<cv::Mat>* input_channels){
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}


class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
};


std::vector<HumanPose> PoseEstimator::extractPoses(
        const std::vector<cv::Mat>& heatMaps,
        const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}


std::vector<HumanPose> PoseEstimator::poseEstimation(const cv::Mat& img){
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);

    ModelOutput model_result = Predict(img); 
    std::vector<HumanPose> poses = Postprocess(model_result, img);
    return poses;
}
} // namespace human_pose_estimation

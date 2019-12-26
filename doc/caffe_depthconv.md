# 模型构造过程 The processes of model construction

1. origin model
source: 
https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth  
ref 
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch   

2. convert torch model to caffe   
ref： https://github.com/ruiminshen/openpose-pytorch.git 

3. add depthwise convolution to caffe model  
[when group==num_output && dilation==1 , modify Covolution to DepthwiseCovolution]  
  
4. compile new caffe[include depthwise convolution function]  
ref: https://github.com/yonghenglh6/DepthwiseConvolution  

















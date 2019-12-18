cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# Define project name
project(poseEstimation)

# 在一些目录中找caffe的配置文件，找到后会将头文件目录设置到${Caffe_INCLUDE_DIRS},将链接库设置到${Caffe_LIBS}中
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

include_directories("/home/xyliu/cpp/caffe/projects/poseEstimation/include")

# # // 单个源文件
add_executable(poseEstimation main.cpp src/pose_estimation.cpp src/peak.cpp src/human_pose.cpp src/render_human_pose.cpp)
target_link_libraries(poseEstimation ${Caffe_LIBRARIES})



#include "opencv2/opencv.hpp"
#include <vector>
#include <map>
#include <string>

#ifndef ZHANG_METHOD_H
#define ZHANG_METHOD_H

using std::vector;

class zhangMethod
{
public:
    zhangMethod();

    void open_yaml(std::string);
    void findCameraPosition();

private:
    void read_point2D(cv::FileStorage &, std::string);
    void read_point3D(cv::FileStorage &, std::string);
    cv::Mat estimateK();
    cv::Mat estimateP();
    cv::Mat estimateRT(cv::Mat &);
    cv::Mat estimateCameraPosition(cv::Mat &);
    cv::Mat addHomography(cv::Mat &);
    cv::Mat add2D3DCorrespondence(cv::Point2f, cv::Mat);
    cv::Mat getOmega(cv::Mat &);
    std::map<std::string, vector<cv::Point2f>> mapPlanePoint2Ds;
    std::map<std::string, vector<cv::Mat>> mapPlanePoint3Ds;
    cv::Mat homographyLeft;
    cv::Mat homographyRight;
    cv::Mat homographyBottom;
};

#endif
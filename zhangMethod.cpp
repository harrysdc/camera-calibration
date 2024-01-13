#include "zhangMethod.hpp"
#include <iostream>
#include <math.h>

void zhangMethod::open_yaml(std::string file)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    read_point2D(fs, "left");
    read_point2D(fs, "right");
    read_point2D(fs, "bottom");
    read_point2D(fs, "physicalLength");

    read_point2D(fs, "correspondence2D");
    read_point3D(fs, "correspondence3D");

    fs.release();
}

void zhangMethod::read_point2D(cv::FileStorage &fs, std::string key)
{
    vector<cv::Point2f> points;
    cv::FileNode fn = fs[key];
    for (auto it = fn.begin(); it != fn.end(); ++it)
    {
        int x = (int)(*it)["x"];
        int y = (int)(*it)["y"];
        points.push_back(cv::Point2f(x, y));
    }

    mapPlanePoint2Ds[key] = points;
}

void zhangMethod::read_point3D(cv::FileStorage &fs, std::string key)
{
    vector<cv::Mat> point3Ds;
    cv::FileNode fn = fs[key];
    for (auto it = fn.begin(); it != fn.end(); ++it)
    {
        cv::Mat point3D(4, 1, CV_64F);
        point3D.at<double>(0, 0) = (double)(*it)["x"];
        point3D.at<double>(1, 0) = (double)(*it)["y"];
        point3D.at<double>(2, 0) = (double)(*it)["z"];
        point3D.at<double>(3, 0) = 1.0;
        point3Ds.push_back(point3D);
    }

    mapPlanePoint3Ds[key] = point3Ds;
}

cv::Mat zhangMethod::estimateK()
{
    // correspondence as reference
    vector<cv::Point2f> correspondence = mapPlanePoint2Ds["physicalLength"];

    // homography of correspondence to image plane
    homographyLeft = cv::findHomography(correspondence, mapPlanePoint2Ds["left"]);
    homographyRight = cv::findHomography(correspondence, mapPlanePoint2Ds["right"]);
    homographyBottom = cv::findHomography(correspondence, mapPlanePoint2Ds["bottom"]);

    // real part = 0 and imaginary part = 0
    cv::Mat matrixA(6, 6, CV_64F);
    cv::Mat row01 = addHomography(homographyLeft);
    cv::Mat row23 = addHomography(homographyRight);
    cv::Mat row45 = addHomography(homographyBottom);
    vector<cv::Mat> mats = {row01, row23, row45};
    cv::vconcat(mats, matrixA);

    // SVD
    cv::Mat w, u, vt;
    cv::SVD::compute(matrixA, w, u, vt);

    // omega
    cv::Mat omega = getOmega(vt);
    cv::Mat omegaInv = omega.inv();
    omegaInv.convertTo(omegaInv, CV_64F, 1.0 / omegaInv.at<double>(2, 2));

    // (c, e, d, b, a)
    double c = omegaInv.at<double>(0, 2);
    double e = omegaInv.at<double>(1, 2);
    double d = sqrt(omegaInv.at<double>(1, 1) - e * e);
    double b = (omegaInv.at<double>(0, 1) - c * e) / d;
    double a = sqrt(omegaInv.at<double>(0, 0) - b * b - c * c);

    // k
    cv::Mat k(3, 3, CV_64F);
    k.at<double>(0, 0) = a;
    k.at<double>(0, 1) = b;
    k.at<double>(0, 2) = c;
    k.at<double>(1, 0) = 0.0;
    k.at<double>(1, 1) = d;
    k.at<double>(1, 2) = e;
    k.at<double>(2, 0) = 0.0;
    k.at<double>(2, 1) = 0.0;
    k.at<double>(2, 2) = 1;

    return k;
}

cv::Mat zhangMethod::addHomography(cv::Mat &h)
{
    cv::Mat a(2, 6, CV_64F);
    double h11 = h.at<double>(0, 0);
    double h12 = h.at<double>(1, 0);
    double h13 = h.at<double>(2, 0);
    double h21 = h.at<double>(0, 1);
    double h22 = h.at<double>(1, 1);
    double h23 = h.at<double>(2, 1);
    a.at<double>(0, 0) = h11 * h21;
    a.at<double>(0, 1) = h11 * h22 + h12 * h21;
    a.at<double>(0, 2) = h11 * h23 + h13 * h21;
    a.at<double>(0, 3) = h12 * h22;
    a.at<double>(0, 4) = h12 * h23 + h13 * h22;
    a.at<double>(0, 5) = h13 * h23;
    a.at<double>(1, 0) = h11 * h11 - h21 * h21;
    a.at<double>(1, 1) = 2 * (h11 * h12 - h21 * h22);
    a.at<double>(1, 2) = 2 * (h11 * h13 - h21 * h23);
    a.at<double>(1, 3) = h12 * h12 - h22 * h22;
    a.at<double>(1, 4) = 2 * (h12 * h13 - h22 * h23);
    a.at<double>(1, 5) = h13 * h13 - h23 * h23;

    return a;
}

cv::Mat zhangMethod::getOmega(cv::Mat &vt)
{
    cv::Mat v = vt.t();
    cv::Mat w(3, 3, CV_64F);
    w.at<double>(0, 0) = v.at<double>(0, 5);
    w.at<double>(0, 1) = v.at<double>(1, 5);
    w.at<double>(0, 2) = v.at<double>(2, 5);
    w.at<double>(1, 0) = v.at<double>(1, 5);
    w.at<double>(1, 1) = v.at<double>(3, 5);
    w.at<double>(1, 2) = v.at<double>(4, 5);
    w.at<double>(2, 0) = v.at<double>(2, 5);
    w.at<double>(2, 1) = v.at<double>(4, 5);
    w.at<double>(2, 2) = v.at<double>(5, 5);

    return w;
}

cv::Mat zhangMethod::estimateRT(cv::Mat &k)
{
    // r1
    cv::Mat r1 = k.inv() * homographyLeft.col(0);
    double norm = sqrt(r1.at<double>(0, 0) * r1.at<double>(0, 0) + r1.at<double>(1, 0) * r1.at<double>(1, 0) + r1.at<double>(2, 0) * r1.at<double>(2, 0));
    r1 = r1 / norm;

    // r2
    cv::Mat r2 = k.inv() * homographyLeft.col(1);
    cv::normalize(r2, r2, 1.0, 0.0, cv::NORM_L2);

    // r3
    cv::Mat r3 = r1.cross(r2);
    cv::normalize(r3, r3, 1.0, 0.0, cv::NORM_L2);

    // t
    cv::Mat t = k.inv() * homographyLeft.col(2);
    t = t / norm;

    cv::Mat rt(3, 4, CV_64F);
    vector<cv::Mat> mats = {r1, r2, r3, t};
    cv::hconcat(mats, rt);

    return rt;
}

cv::Mat zhangMethod::estimateP()
{

    cv::Mat puvmatrix(12, 12, CV_64F);
    vector<cv::Mat> mats;
    vector<cv::Mat> point3Ds = mapPlanePoint3Ds["correspondence3D"];
    vector<cv::Point2f> point2Ds = mapPlanePoint2Ds["correspondence2D"];
    for (int i = 0; i < 7; i++)
    {
        cv::Mat rows(2, 14, CV_64F);
        rows = add2D3DCorrespondence(point2Ds.at(i), point3Ds.at(i)); // a pair gives two equations.
        mats.push_back(rows);
    }
    cv::vconcat(mats, puvmatrix);

    // SVD
    cv::Mat w, u, vt, v;
    cv::SVD::compute(puvmatrix, w, u, vt);
    v = vt.t();

    // getP
    cv::Mat p(3, 4, CV_64F);
    p.row(0) = v(cv::Rect(11, 0, 1, 4)).t();
    p.row(1) = v(cv::Rect(11, 4, 1, 4)).t();
    p.row(2) = v(cv::Rect(11, 8, 1, 4)).t();
    double norm = p.at<double>(2, 3);
    p = p / norm;

    return p;
}

cv::Mat zhangMethod::estimateCameraPosition(cv::Mat &rt)
{
    cv::Mat cameraPosition(3, 1, CV_64F);
    cv::Mat r = rt(cv::Rect(0, 0, 3, 3));
    cv::Mat t = rt.col(3);
    cameraPosition = r.t() * (-1 * t);

    return cameraPosition;
}

cv::Mat zhangMethod::add2D3DCorrespondence(cv::Point2f point2D, cv::Mat point3D)
{
    cv::Mat puvmatrixRows = cv::Mat::zeros(cv::Size(12, 2), CV_64F);
    puvmatrixRows(cv::Rect(0, 0, 4, 1)) = point3D.t();
    puvmatrixRows(cv::Rect(4, 1, 4, 1)) = point3D.t();
    puvmatrixRows(cv::Rect(8, 0, 4, 1)) = -1.0 * point2D.x * point3D.t();
    puvmatrixRows(cv::Rect(8, 1, 4, 1)) = -1.0 * point2D.y * point3D.t();
    return puvmatrixRows;
}

zhangMethod::zhangMethod()
{
}

void zhangMethod::findCameraPosition()
{
    cv::Mat k = estimateK(); // 3x3
    cv::Mat p = estimateP(); // 3x4
    cv::Mat rt = k.inv() * p;
    double norm = sqrt(rt.at<double>(0, 0) * rt.at<double>(0, 0) + rt.at<double>(1, 0) * rt.at<double>(1, 0) + rt.at<double>(2, 0) * rt.at<double>(2, 0));
    rt = rt / norm;

    cv::Mat cameraPosition = estimateCameraPosition(rt);

    std::cout << "k" << std::endl
              << k << std::endl
              << std::endl;
    std::cout << "rt" << std::endl
              << rt << std::endl
              << std::endl;
    std::cout << "camera position" << std::endl
              << cameraPosition << std::endl;
}
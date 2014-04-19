#ifndef FOREGROUNDEXTRACTOR_H
#define FOREGROUNDEXTRACTOR_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

class ForegroundExtractor
{
public:
    ForegroundExtractor();

    void setIternum(int num);

    cv::Mat getForegroundMask(const cv::Mat &src, const cv::Rect &roi);
    cv::Mat getForeground(const cv::Mat &src, const cv::Mat &foreMask);
    cv::Mat getForegroundWeighted(const cv::Mat &src, const cv::Mat &foreMask, const double weight);

private:
    int iternum;
};

#endif // FOREGROUNDEXTRACTOR_H

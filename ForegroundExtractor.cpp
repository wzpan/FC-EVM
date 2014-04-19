#include "ForegroundExtractor.h"

ForegroundExtractor::ForegroundExtractor () : iternum(5)
{
}

void ForegroundExtractor::setIternum(int num)
{
    iternum = num;
}

cv::Mat ForegroundExtractor::getForegroundMask(const cv::Mat &src, const cv::Rect &roi)
{
    cv::Mat bgModel, fgModel, dst;	// the models (internally used)

    // GrabCut segmentation
    cv::grabCut(src,	// input image
                dst,	// segmentation result
                roi,  // rectangle contain foreground
                bgModel, fgModel,  // models
                iternum,		// number of iterations
                cv::GC_INIT_WITH_RECT	// use rectangle
                );

    // Get the pixels marked as likely foreground
    cv::compare(dst, cv::GC_PR_FGD, dst, cv::CMP_EQ);
    return dst;
}


cv::Mat ForegroundExtractor::getForeground(const cv::Mat &src, const cv::Mat &foreMask)
{
    // Generate output image
    cv::Mat foreground(src.size(), CV_8UC3,
                    cv::Scalar(255, 255, 255));
    src.copyTo(foreground,	// bg pixels are not copied
                foreMask);

    return foreground;
}

cv::Mat ForegroundExtractor::getForegroundWeighted(const cv::Mat &src, const cv::Mat &foreMask, const double weight)
{
    // Generate output image
    cv::Mat foreground(src.size(), CV_8UC3,
                    cv::Scalar(255, 255, 255));
    src.copyTo(foreground,	// bg pixels are not copied
                foreMask);

    cv::addWeighted(foreground, 1.0-weight, src, weight, 0.0, foreground);

    return foreground;
}

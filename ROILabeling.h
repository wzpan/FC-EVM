#ifndef ROILABELING_H
#define ROILABELING_H

#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"

static bool selectObject = false;
static int trackObject = 0;
static cv::Rect selection;
static int frameWidth, frameHeight;
static cv::Point origin;
static int sliceX = -1;
static int sliceY = -1;

static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= cv::Rect(0, 0, frameWidth, frameHeight);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = cv::Point(x,y);
        selection = cv::Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}

static void onMouseX( int event, int x, int y, int, void* )
{
    if( CV_EVENT_LBUTTONDOWN == event )
    {
        sliceX = x;
    }
}

static void onMouseY( int event, int x, int y, int, void* )
{
    if( CV_EVENT_LBUTTONDOWN == event )
    {
        sliceY = y;
    }
}

#endif // ROILABELING_H

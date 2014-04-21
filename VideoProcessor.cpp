// Yet anther C++ implementation of EVM, based on OpenCV and Qt.
// Copyright (C) 2014  Joseph Pan <cs.wzpan@gmail.com>
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301 USA
// 


#include "VideoProcessor.h"

VideoProcessor::VideoProcessor(QObject *parent)
  : QObject(parent)
  , delay(-1)
  , rate(0)
  , fnumber(0)
  , length(0)
  , stop(true)
  , modify(false)
  , curPos(0)
  , curIndex(0)
  , curLevel(0)
  , digits(0)
  , extension(".avi")
  , levels(6)
  , alpha(10)
  , lambda_c(80)
  , fl(0.05)
  , fh(0.4)
  , chromAttenuation(0.1)
  , delta(0)
  , exaggeration_factor(0.2)
  , lambda(0)
  , vmin(10)
  , vmax(256)
  , smin(30)
  , isTrack(false)
  , isGrabcut(false)  
  , isKalman(false)
  , isLabeled(false)
  , showROI(false)
  , roi(cv::Rect(0,0,0,0))
{
    connect(this, SIGNAL(revert()), this, SLOT(revertVideo()));
}

/** 
 * setDelay	-	 set a delay between each frame
 *
 * 0 means wait at each frame, 
 * negative means no delay
 * @param d	-	delay param
 */
void VideoProcessor::setDelay(int d)
{
    delay = d;
}

/** 
 * getNumberOfProcessedFrames	-	a count is kept of the processed frames
 *
 *
 * @return the number of processed frames
 */
long VideoProcessor::getNumberOfProcessedFrames()
{
    return fnumber;
}

/** 
 * getNumberOfPlayedFrames	-	get the current playing progress
 *
 * @return the number of played frames
 */
long VideoProcessor::getNumberOfPlayedFrames()
{
    return curPos;
}

/** 
 * getFrameSize	-	return the size of the video frame
 *
 *
 * @return the size of the video frame
 */
cv::Size VideoProcessor::getFrameSize()
{
    int w = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    return cv::Size(w,h);
}

/** 
 * getFrameNumber	-	return the frame number of the next frame
 *
 *
 * @return the frame number of the next frame
 */
long VideoProcessor::getFrameNumber()
{
    long f = static_cast<long>(capture.get(CV_CAP_PROP_POS_FRAMES));

    return f;
}

/** 
 * getPositionMS	-	return the position in milliseconds
 *
 * @return the position in milliseconds
 */
double VideoProcessor::getPositionMS()
{
    double t = capture.get(CV_CAP_PROP_POS_MSEC);

    return t;
}

/** 
 * getFrameRate	-	return the frame rate
 *
 *
 * @return the frame rate
 */
double VideoProcessor::getFrameRate()
{
    double r = capture.get(CV_CAP_PROP_FPS);

    return r;
}

/** 
 * getLength	-	return the number of frames in video
 *
 * @return the number of frames
 */
long VideoProcessor::getLength()
{
    return length;
}


/** 
 * getLengthMS	-	return the video length in milliseconds
 *
 *
 * @return the length of length in milliseconds
 */
double VideoProcessor::getLengthMS()
{
    double l = 1000.0 * length / rate;
    return l;
}

bool VideoProcessor::getFirstFrame(cv::Mat &frame)
{
    long pos = curPos;
    jumpTo(0);
    if (getNextFrame(frame)){
        jumpTo(pos);
        return true;
    }
    else
        return false;
}

/**
 * calculateLength	-	recalculate the number of frames in video
 *
 * normally doesn't need it unless getLength()
 * can't return a valid value
 */
void VideoProcessor::calculateLength()
{
    long l = 0;
    cv::Mat img;
    cv::VideoCapture tempCapture(tempFile);
    while(tempCapture.read(img)){
        ++l;
    }
    length = l;
    tempCapture.release();
}

/** 
 * spatialFilter	-	spatial filtering an image
 *
 * @param src		-	source image 
 * @param pyramid	-	destinate pyramid
 */
bool VideoProcessor::spatialFilter(const cv::Mat &src, std::vector<cv::Mat> &pyramid)
{
    switch (spatialType) {
    case LAPLACIAN:     // laplacian pyramid
        return buildLaplacianPyramid(src, levels, pyramid);
        break;
    case GAUSSIAN:      // gaussian pyramid
        return buildGaussianPyramid(src, levels, pyramid);
        break;
    default:
        return false;
        break;
    }
}

/** 
 * temporalFilter	-	temporal filtering an image
 *
 * @param src	-	source image
 * @param dst	-	destinate image
 */
void VideoProcessor::temporalFilter(const cv::Mat &src,
                                    cv::Mat &dst)
{
    switch(temporalType) {
    case IIR:       // IIR bandpass filter
        temporalIIRFilter(src, dst);
        break;
    case IDEAL:     // Ideal bandpass filter
        temporalIdealFilter(src, dst);
        break;
    default:        
        break;
    }
    return;
}

/** 
 * temporalIIRFilter	-	temporal IIR filtering an image
 *                          (thanks to Yusuke Tomoto)
 * @param pyramid	-	source image
 * @param filtered	-	filtered result
 *
 */
void VideoProcessor::temporalIIRFilter(const cv::Mat &src,
                                    cv::Mat &dst)
{
    cv::Mat temp1 = (1-fh)*lowpass1[curLevel] + fh*src;
    cv::Mat temp2 = (1-fl)*lowpass2[curLevel] + fl*src;
    lowpass1[curLevel] = temp1;
    lowpass2[curLevel] = temp2;
    dst = lowpass1[curLevel] - lowpass2[curLevel];
}

/** 
 * temporalIdalFilter	-	temporal IIR filtering an image pyramid of concat-frames
 *                          (Thanks to Daniel Ron & Alessandro Gentilini)
 *
 * @param pyramid	-	source pyramid of concatenate frames
 * @param filtered	-	concatenate filtered result
 *
 */
void VideoProcessor::temporalIdealFilter(const cv::Mat &src,
                                          cv::Mat &dst)
{
    cv::Mat channels[3];

    // split into 3 channels
    cv::split(src, channels);

    for (int i = 0; i < 3; ++i){

        cv::Mat current = channels[i];  // current channel
        cv::Mat tempImg;

        int width = cv::getOptimalDFTSize(current.cols);
        int height = cv::getOptimalDFTSize(current.rows);

        cv::copyMakeBorder(current, tempImg,
                           0, height - current.rows,
                           0, width - current.cols,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        // do the DFT
        cv::dft(tempImg, tempImg, cv::DFT_ROWS | cv::DFT_SCALE);

        // construct the filter
        cv::Mat filter = tempImg.clone();
        createIdealBandpassFilter(filter, fl, fh, rate);

        // apply filter
        cv::mulSpectrums(tempImg, filter, tempImg, cv::DFT_ROWS);

        // do the inverse DFT on filtered image
        cv::idft(tempImg, tempImg, cv::DFT_ROWS | cv::DFT_SCALE);

        // copy back to the current channel
        tempImg(cv::Rect(0, 0, current.cols, current.rows)).copyTo(channels[i]);
    }
    // merge channels
    cv::merge(channels, 3, dst);

    // normalize the filtered image
    cv::normalize(dst, dst, 0, 1, CV_MINMAX);
}

/** 
 * amplify	-	ampilfy the motion
 *
 * @param filtered	- motion image
 */
void VideoProcessor::amplify(const cv::Mat &src, cv::Mat &dst)
{
    float curAlpha;
    switch (spatialType) {
    case LAPLACIAN:        
        //compute modified alpha for this level
        curAlpha = lambda/delta/8 - 1;
        curAlpha *= exaggeration_factor;
        if (curLevel==levels || curLevel==0)     // ignore the highest and lowest frequency band
            dst = src * 0;
        else
            dst = src * cv::min(alpha, curAlpha);
        break;
    case GAUSSIAN:
        dst = src * alpha;
        break;
    default:
        break;
    }
}

/** 
 * attenuate	-	attenuate I, Q channels
 *
 * @param src	-	source image
 * @param dst   -   destinate image
 */
void VideoProcessor::attenuate(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat planes[3];
    cv::split(src, planes);
    planes[1] = planes[1] * chromAttenuation;
    planes[2] = planes[2] * chromAttenuation;
    cv::merge(planes, 3, dst);
}


/** 
 * concat	-	concat all the frames into a single large Mat
 *              where each column is a reshaped single frame
 *
 * @param frames	-	frames of the video sequence
 * @param dst		-	destinate concatnate image
 */
void VideoProcessor::concat(const std::vector<cv::Mat> &frames,
                            cv::Mat &dst)
{
    cv::Size frameSize = frames.at(0).size();
    cv::Mat temp(frameSize.width*frameSize.height, length-1, CV_32FC3);
    for (int i = 0; i < length-1; ++i) {
        // get a frame if any
        cv::Mat input = frames.at(i);
        // reshape the frame into one column
        cv::Mat reshaped = input.reshape(3, input.cols*input.rows).clone();
        cv::Mat line = temp.col(i);
        // save the reshaped frame to one column of the destinate big image
        reshaped.copyTo(line);
    }
    temp.copyTo(dst);
}

/**
 * deConcat	-	de-concat the concatnate image into frames
 *
 * @param src       -   source concatnate image
 * @param framesize	-	frame size
 * @param frames	-	destinate frames
 */
void VideoProcessor::deConcat(const cv::Mat &src,
                              const cv::Size &frameSize,
                              std::vector<cv::Mat> &frames)
{
    for (int i = 0; i < length-1; ++i) {    // get a line if any
        cv::Mat line = src.col(i).clone();
        cv::Mat reshaped = line.reshape(3, frameSize.height).clone();
        frames.push_back(reshaped);
    }
}

/**
 * createIdealBandpassFilter	-	create a 1D ideal band-pass filter
 *
 * @param filter    -	destinate filter
 * @param fl        -	low cut-off
 * @param fh		-	high cut-off
 * @param rate      -   sampling rate(i.e. video frame rate)
 */
void VideoProcessor::createIdealBandpassFilter(cv::Mat &filter, double fl, double fh, double rate)
{
    int width = filter.cols;
    int height = filter.rows;

    fl = 2 * fl * width / rate;
    fh = 2 * fh * width / rate;

    double response;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // filter response
            if (j >= fl && j <= fh)
                response = 1.0f;
            else
                response = 0.0f;
            filter.at<float>(i, j) = response;
        }
    }
}

void VideoProcessor::printMat(cv::Mat &image)
{
    int nl = image.rows;		// number of lines
    int nc = image.cols * image.channels();

    // this loop is executed only once
    for (int j=0; j<nl; ++j) {
        uchar *data = image.ptr<uchar>(j);
        for (int i=0; i<nc; ++i) {
            // process each pixel
            printf("%d", data[i]);
            // end of pixel processing
        }
        printf("\n");
    }
}

void VideoProcessor::shiftDFT(cv::Mat &fImage)
{
    cv::Mat tmp, q0, q1, q2, q3;

    // first crop the image, if it has an odd number of rows or columns

    fImage = fImage(cv::Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

    int cx = fImage.cols / 2;
    int cy = fImage.rows / 2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center

    q0 = fImage(cv::Rect(0, 0, cx, cy));
    q1 = fImage(cv::Rect(cx, 0, cx, cy));
    q2 = fImage(cv::Rect(0, cy, cx, cy));
    q3 = fImage(cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat VideoProcessor::LaplacianBlend(const Mat &l, const Mat &r, const cv::Mat_<float> &m)
{
    LaplacianBlending lb(l,r,m,levels);
    return lb.blend();
}

/** 
 * getCodec	-	get the codec of input video
 *
 * @param codec	-	the codec arrays
 *
 * @return the codec integer
 */
int VideoProcessor::getCodec(char codec[])
{
    union {
        int value;
        char code[4]; } returned;

    returned.value = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));

    codec[0] = returned.code[0];
    codec[1] = returned.code[1];
    codec[2] = returned.code[2];
    codec[3] = returned.code[3];

    return returned.value;
}


/** 
 * getTempFile	-	temp file lists
 *
 * @param str	-	the reference of the output string
 */
void VideoProcessor::getTempFile(std::string &str)
{
    if (!tempFileList.empty()){
        str = tempFileList.back();
        tempFileList.pop_back();
    } else {
        str = "";
    }
}

/** 
 * getCurTempFile	-	get current temp file
 *
 * @param str	-	the reference of the output string
 */
void VideoProcessor::getCurTempFile(std::string &str)
{
    str = tempFile;
}

/** 
 * setInput	-	set the name of the expected video file
 *
 * @param fileName	-	the name of the video file
 *
 * @return True if success. False otherwise
 */
bool VideoProcessor::setInput(const std::string &fileName)
{
    fnumber = 0;
    tempFile = fileName;

    // In case a resource was already
    // associated with the VideoCapture instance
    if (isOpened()){
        capture.release();
    }

    // Open the video file
    if(capture.open(fileName)){
        // read parameters
        calculateLength();
        rate = getFrameRate();
        cv::Mat frame;
        if (getFirstFrame(frame)){
            emit showFrame(frame);
            emit updateBtn();
            return true;
        } else
            return false;
    } else {
        return false;
    }
}

/** 
 * setOutput	-	set the output video file
 *
 * by default the same parameters than input video will be used
 *
 * @param filename	-	filename prefix
 * @param codec		-	the codec
 * @param framerate -	frame rate
 * @param isColor	-	is the video colorful
 *
 * @return True if successful. False otherwise
 */
bool VideoProcessor::setOutput(const std::string &filename, int codec, double framerate, bool isColor)
{
    outputFile = filename;
    extension.clear();

    if (framerate==0.0)
        framerate = getFrameRate(); // same as input

    char c[4];
    // use same codec as input
    if (codec==0) {
        codec = getCodec(c);
    }

    // Open output video
    return writer.open(outputFile, // filename
                       codec, // codec to be used
                       framerate,      // frame rate of the video
                       getFrameSize(), // frame size
                       isColor);       // color video?
}

/** 
 * set the output as a series of image files
 *
 * extension must be ".jpg", ".bmp" ...
 *
 * @param filename	-	filename prefix
 * @param ext		-	image file extension
 * @param numberOfDigits	-	number of digits
 * @param startIndex	-	start index
 *
 * @return True if successful. False otherwise
 */
bool VideoProcessor::setOutput(const std::string &filename, const std::string &ext, int numberOfDigits, int startIndex)
{
    // number of digits must be positive
    if (numberOfDigits<0)
        return false;

    // filenames and their common extension
    outputFile = filename;
    extension = ext;

    // number of digits in the file numbering scheme
    digits = numberOfDigits;
    // start numbering at this index
    curIndex = startIndex;

    return true;
}

/** 
 * setTemp	-	set the temp video file
 *
 * by default the same parameters to the input video
 *
 * @param codec	-	video codec
 * @param framerate	-	frame rate
 * @param isColor	-	is the video colorful
 *
 * @return True if successful. False otherwise
 */
bool VideoProcessor::createTemp(cv::Size frameSize)
{
    std::stringstream ss;
    ss << "temp_" << QDateTime::currentDateTime().toTime_t() << ".avi";
    tempFile = ss.str();

    tempFileList.push_back(tempFile);

    double framerate = getFrameRate(); // same as input

    // Open output video
    return tempWriter.open(tempFile, // filename
                       CV_FOURCC('M', 'J', 'P', 'G'), // codec to be used
                       framerate,      // frame rate of the video
                       frameSize, // frame size
                       true);       // color video?
}

/** 
 * setSpatialFilter	-	set the spatial filter
 *
 * @param type	-	spatial filter type. Could be:
 *					1. LAPLACIAN: laplacian pyramid
 *					2. GAUSSIAN: gaussian pyramid
 */
void VideoProcessor::setSpatialFilter(spatialFilterType type)
{
    spatialType = type;
}

/** 
 * setTemporalFilter	-	set the temporal filter
 *
 * @param type	-	temporal filter type. Could be:
 *					1. IIR: second order(IIR) filter
 *					2. IDEAL: ideal bandpass filter
 */
void VideoProcessor::setTemporalFilter(temporalFilterType type)
{
    temporalType = type;
}

/** 
 * stopIt	-	stop playing or processing
 *
 */
void VideoProcessor::stopIt()
{
    stop = true;
    emit revert();
}

/** 
 * prevFrame	-	display the prev frame of the sequence
 *
 */
void VideoProcessor::prevFrame()
{
    if(isStop())
        pauseIt();
    if (curPos >= 0){
        curPos -= 1;
        jumpTo(curPos);
    }
    emit updateProgressBar();
}

/** 
 * nextFrame	-	display the next frame of the sequence
 *
 */
void VideoProcessor::nextFrame()
{
    if(isStop())
        pauseIt();
    curPos += 1;
    if (curPos <= length){
        curPos += 1;
        jumpTo(curPos);
    }
    emit updateProgressBar();
}

/** 
 * jumpTo	-	Jump to a position
 *
 * @param index	-	frame index
 *
 * @return True if success. False otherwise
 */
bool VideoProcessor::jumpTo(long index)
{
    if (index >= length){
        return 1;
    }

    cv::Mat frame;
    bool re = capture.set(CV_CAP_PROP_POS_FRAMES, index);

    if (re){
        capture.read(frame);
        emit showFrame(frame);
    }

    return re;
}


/** 
 * jumpToMS	-	jump to a position at a time
 *
 * @param pos	-	time
 *
 * @return True if success. False otherwise
 *
 */
bool VideoProcessor::jumpToMS(double pos)
{
    return capture.set(CV_CAP_PROP_POS_MSEC, pos);
}


/** 
 * close	-	close the video
 *
 */
void VideoProcessor::close()
{
    rate = 0;
    length = 0;
    modify = 0;
    capture.release();
    writer.release();
    tempWriter.release();
}

/** 
 * setROI	-	set the labeled ROI for mean-shift and GrabCut
 *
 * @param roi   -   rectangle of roi
 */
void VideoProcessor::setROI(cv::Rect roi)
{
    this->roi = roi;

    if (roi.area()) {
        // extract ROI image
        cv::Mat firstFrame;
        getFirstFrame(firstFrame);
        imageROI = firstFrame(roi);
        showFrame(firstFrame);

        isLabeled = true;
    } else {
        isLabeled = false;
    }
}

/** 
 * isStop	-	Is the processing stop
 *
 *
 * @return True if not processing/playing. False otherwise
 */
bool VideoProcessor::isStop()
{
    return stop;
}

/** 
 * isModified	-	Is the video modified?
 *
 *
 * @return True if modified. False otherwise
 */
bool VideoProcessor::isModified()
{
    return modify;
}

/** 
 * isOpened	-	Is the player opened?
 *
 *
 * @return True if opened. False otherwise 
 */
bool VideoProcessor::isOpened()
{
    return capture.isOpened();
}

/** 
 * getNextFrame	-	get the next frame if any
 *
 * @param frame	-	the expected frame
 *
 * @return True if success. False otherwise
 */
bool VideoProcessor::getNextFrame(cv::Mat &frame)
{
    return capture.read(frame);
}

Mat VideoProcessor::extractXTSlice(int y)
{
    cv::Mat input;
    cv::Size frameSize = getFrameSize();
    assert(y <= frameSize.height);
    cv::Mat dst(length-1, frameSize.width, CV_8UC3);
    long pos = curPos;
    jumpTo(0);
    int i = 0;
    while (getNextFrame(input)) {
        // reshape the frame into one column
        cv::Mat row = input.row(y).clone();
        cv::Mat line = dst.row(i++);
        // save the reshaped frame to one column of the destinate big image
        row.copyTo(line);
    }
    jumpTo(pos);

    return dst;
}

cv::Mat VideoProcessor::extractYTSlice(int x)
{
    cv::Mat input;
    cv::Size frameSize = getFrameSize();
    assert(x <= frameSize.width);
    cv::Mat dst(frameSize.height, length-1, CV_8UC3);
    long pos = curPos;
    jumpTo(0);
    int i = 0;
    while (getNextFrame(input)) {
        // reshape the frame into one column
        cv::Mat column = input.col(x).clone();
        cv::Mat line = dst.col(i++);
        // save the reshaped frame to one column of the destinate big image
        column.copyTo(line);
    }
    jumpTo(pos);

    return dst;
}

/** 
 * writeNextFrame	-	to write the output frame
 *
 * @param frame	-	the frame to be written
 */
void VideoProcessor::writeNextFrame(cv::Mat &frame)
{
    if (extension.length()) { // then we write images

        std::stringstream ss;
        ss << outputFile << std::setfill('0') << std::setw(digits) << curIndex++ << extension;
        cv::imwrite(ss.str(),frame);

    } else { // then write video file

        writer.write(frame);
    }
}

/** 
 * playIt	-	play the frames of the sequence
 *
 */
void VideoProcessor::playIt()
{
    // current frame
    cv::Mat input;

    // if no capture device has been set
    if (!isOpened())
        return;

    // is playing
    stop = false;

    // update buttons
    emit updateBtn();

    while (!isStop()) {

        // read next frame if any
        if (!getNextFrame(input))
            break;

        curPos = capture.get(CV_CAP_PROP_POS_FRAMES);

        // display input frame
        emit showFrame(input);

        // update the progress bar
        emit updateProgressBar();

        // introduce a delay
        emit sleep(delay);
    }
    if (!isStop()){
        emit revert();
    }
}

/** 
 * pauseIt	-	pause playing
 *
 */
void VideoProcessor::pauseIt()
{
    stop = true;
    emit updateBtn();
}

/** 
 * motionMagnify	-	eulerian motion magnification
 *
 */
void VideoProcessor::motionMagnify()
{
    // set filter
    setSpatialFilter(LAPLACIAN);
    setTemporalFilter(IIR);

    // create a temp file
    createTemp(getFrameSize());

    // current frame
    cv::Mat input;
    // output frame
    cv::Mat output;
    // temp frame
    cv::Mat original, temp;

    // kalman filter
    cv::KalmanFilter KF(4, 2, 0);
    cv::Mat_<float> measurement(2, 1);
    measurement.setTo(cv::Scalar(0));

    // foreground for grabcut
    cv::Mat foreMask;

    ForegroundExtractor *extractor = new ForegroundExtractor();
    extractor->setIternum(1);

    // for meanshift
    cv::Mat hsv, hue, mask, hist, histimg, backproj;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

    // feature video
    double frameWidth, frameHeight;
    double featureWidth = 0;
    double featureHeight = 0;
    cv::Size featureSize;
    cv::Mat featureBefore, featureAfter;

    // motion image
    cv::Mat motion;

    std::vector<cv::Mat> pyramid;
    std::vector<cv::Mat> filtered;

    // if no capture device has been set
    if (!isOpened())
        return;

    // set the modify flag to be true
    modify = true;

    // is processing
    stop = false;

    // save the current position
    long pos = curPos;    

    // jump to the first frame
    jumpTo(0);

    while (!isStop()) {

        // read next frame if any
        if (!getNextFrame(input))
            break;        

        temp = input.clone();

        // search ROI using mean-shift
        if (isLabeled && isTrack) {

            // Convert to HSV space
            input.convertTo(hsv, CV_BGR2HSV);

            int _vmin = vmin, _vmax = vmax;
            cv::inRange(hsv, cv::Scalar(0, smin, MIN(_vmin,_vmax)),
                    cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

            // init meanshift
            if (fnumber == 0) {
                histimg = cv::Mat::zeros(input.size(), CV_8UC3);

                cv::Mat myroi(hue, roi), maskroi(mask, roi);
                cv::calcHist(&myroi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                cv::normalize(hist, hist, 0, 255, CV_MINMAX);

                histimg = cv::Scalar::all(0);
                int binW = histimg.cols / hsize;
                cv::Mat buf(1, hsize, CV_8UC3);
                for( int i = 0; i < hsize; i++ )
                    buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180./hsize), 255, 255);
                cvtColor(buf, buf, CV_HSV2BGR);

                for( int i = 0; i < hsize; i++ )
                {
                    int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                    cv::rectangle( histimg, cv::Point(i*binW,histimg.rows),
                                   cv::Point((i+1)*binW,histimg.rows - val),
                                   cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8 );
                }

                // feature video size
                frameWidth = getFrameSize().width;
                frameHeight = getFrameSize().height;
                featureHeight = frameHeight / 2.0;
                featureWidth = roi.width * featureHeight / roi.height;
                featureSize = cv::Size(featureWidth, featureHeight);

                createTemp(cv::Size(frameWidth + featureWidth, frameHeight));
            }

            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            backproj &= mask;
            cv::meanShift(backproj, roi, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1 ));

            // kalman filtering
            if (isKalman) {
                if (fnumber == 0) {
                    KF.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);

                    // init kalman filter
                    KF.statePre.at<float>(0) = roi.x;
                    KF.statePre.at<float>(1) = roi.y;
                    KF.statePre.at<float>(2) = 0;
                    KF.statePre.at<float>(3) = 0;

                    setIdentity(KF.measurementMatrix);
                    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
                    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
                    setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

                    cv::Mat estimated = KF.correct(measurement);
                    cv::Point statePt(estimated.at<float>(0), estimated.at<float>(1));
                } else {
                    // kalman filtering roi position
                    cv::Mat prediction = KF.predict();
                    cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

                    // Get roi position
                    measurement(0) = roi.x;
                    measurement(1) = roi.y;

                    // The "correct" phase
                    cv::Mat estimated = KF.correct(measurement);
                    cv::Point statePt(estimated.at<float>(0), estimated.at<float>(1));

                    roi.x = statePt.x;
                    roi.y = statePt.y;

                }
            }

            temp = input(roi).clone();
            cv::resize(temp, featureBefore, featureSize);

            if (isGrabcut) {
                // calculate foreground mask
                foreMask = extractor->getForegroundMask(input, roi);
            }
        }

        input.convertTo(original, CV_32FC3, 1.0/255.0f);
        temp.convertTo(temp, CV_32FC3, 1.0/255.0f);

        // convert to Lab color space
        cv::cvtColor(temp, temp, CV_BGR2Lab);

        // spatial filtering one frame
        spatialFilter(temp, pyramid);

        // temporal filtering one frame's pyramid
        // and amplify the motion
        if (fnumber == 0){      // is first frame
            lowpass1 = pyramid;
            lowpass2 = pyramid;
            filtered = pyramid;
        } else {
            for (int i=0; i<levels; ++i) {
                curLevel = i;
                temporalFilter(pyramid.at(i), filtered.at(i));
            }

            // amplify each spatial frequency bands
            // according to Figure 6 of paper
            cv::Size filterSize = filtered.at(0).size();
            int w = filterSize.width;
            int h = filterSize.height;

            delta = lambda_c/8.0/(1.0+alpha);
            // the factor to boost alpha above the bound
            // (for better visualization)
            exaggeration_factor = 2.0;

            // compute the representative wavelength lambda
            // for the lowest spatial frequency band of Laplacian pyramid
            lambda = sqrt(w*w + h*h)/3;  // 3 is experimental constant

            for (int i=levels; i>=0; i--) {
                curLevel = i;

                amplify(filtered.at(i), filtered.at(i));

                // go one level down on pyramid
                // representative lambda will reduce by factor of 2
                lambda /= 2.0;
            }
        }

        std::vector<cv::Mat>::iterator it = filtered.begin();
        std::vector<cv::Mat>::iterator itend = filtered.end();

        // output filtered pyramid
        if (fnumber == 0){
            int j = 0;
            std::stringstream ss;
            for(; it < itend; ++it){
                ss.str("");
                ss << "lever-" << j << ".png";
                cv::imwrite(ss.str(), *it);
                ++j;
            }
        }

        // reconstruct motion image from filtered pyramid
        reconImgFromLaplacianPyramid(filtered, levels, motion);

        // attenuate I, Q channels
        attenuate(motion, motion);

        // combine source frame and motion image
        if (fnumber > 0)    // don't amplify first frame
            temp += motion;

        // convert back to rgb color space and CV_8UC3
        output = temp.clone();
        cv::cvtColor(output, output, CV_Lab2BGR);
        output.convertTo(output, CV_8UC3, 255.0, 1.0/255.0);        

        // copy back the roi image to original frame
        if (isLabeled && isTrack) {
            cv::Mat result = cv::Mat(input, roi);
            output.copyTo(result);
            output = input;

            // draw the searched result if neccessary
            if (showROI)
                cv::rectangle(output, roi, cv::Scalar(0,255,0));
        }

        // multi-resolution blending input and output
        if (isLabeled && isGrabcut) {
            cv::Mat foreground, background;
            background = original;
            output.convertTo(foreground, CV_32F, 1.0/255.0f);

            cv::Mat_<float> mask;
            foreMask.convertTo(mask, CV_32F, 1.0/255.0f);

            output = LaplacianBlend(foreground, background, mask);
            output.convertTo(output, CV_8UC3, 255.0, 1.0/255.0);
        }

        // concat the feature image
        if (isLabeled && isTrack) {
            featureAfter = output(roi).clone();
            cv::resize(featureAfter, featureAfter, featureSize);

            cv::Mat resultFrame(cv::Size(frameWidth + featureWidth, frameHeight), CV_8UC3, cv::Scalar(255, 255, 255));
            cv::Mat resultROI;
            resultROI = cv::Mat(resultFrame, cv::Rect(0, 0, frameWidth, frameHeight));
            output.copyTo(resultROI);
            resultROI = cv::Mat(resultFrame, cv::Rect(frameWidth, 0, featureWidth, featureHeight));
            featureBefore.copyTo(resultROI);
            resultROI = cv::Mat(resultFrame, cv::Rect(frameWidth, featureHeight, featureWidth, featureHeight));
            featureAfter.copyTo(resultROI);
            output = resultFrame.clone();
        }

        // write the frame to the temp file
        tempWriter.write(output);

        // update process
        std::string msg= "Processing...";
        emit updateProcessProgress(msg, floor((fnumber++) * 100.0 / length));
    }
    if (!isStop()){
        emit revert();
    }
    emit closeProgressDialog();


    // release the temp writer
    tempWriter.release();

    // change the video to the processed video 
    setInput(tempFile);

    // jump back to the original position
    jumpTo(pos);
}

/**
 * colorMagnify	-	color magnification
 *
 */
void VideoProcessor::colorMagnify()
{
    // set filter
    setSpatialFilter(GAUSSIAN);
    setTemporalFilter(IDEAL);

    // create a temp file
    createTemp(getFrameSize());

    // current frame
    cv::Mat input;
    // output frame
    cv::Mat output;
    // motion image
    cv::Mat motion;

    // temp image
    cv::Mat  temp;

    // roi at each frame
    std::vector<cv::Rect> rois;

    // video frames
    std::vector<cv::Mat> frames, windows;
    // down-sampled frames
    std::vector<cv::Mat> downSampledFrames;
    // filtered frames
    std::vector<cv::Mat> filteredFrames;

    // concatenate image of all the down-sample frames
    cv::Mat_<cv::Vec3f> videoMat;
    // concatenate filtered image
    cv::Mat filtered;

    // foreground for grabcut
    cv::Mat foreMask;
    std::vector<cv::Mat> foreMasks;

    ForegroundExtractor *extractor = new ForegroundExtractor();
    extractor->setIternum(1);

    // kalman filter
    cv::KalmanFilter KF(4, 2, 0);
    cv::Mat_<float> measurement(2, 1);
    measurement.setTo(cv::Scalar(0));

    // for meanshift
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

    cv::Mat hsv, hue, mask, hist, histimg, backproj;

    // feature video
    double frameWidth, frameHeight;
    double featureWidth = 0;
    double featureHeight = 0;
    cv::Size featureSize;
    cv::Mat featureBefore, featureAfter;
    std::vector<cv::Mat> featureBefores;

    // if no capture device has been set
    if (!isOpened())
        return;

    // set the modify flag to be true
    modify = true;

    // is processing
    stop = false;

    // save the current position
    long pos = curPos;

    // jump to the first frame
    jumpTo(0);

    // 1. spatial filtering
    while (getNextFrame(input) && !isStop()) {

        frames.push_back(input.clone());

        temp = input.clone();

        // search ROI using mean-shift
        if (isLabeled && isTrack) {

            // Convert to HSV space
            input.convertTo(hsv, CV_BGR2HSV);

            int _vmin = vmin, _vmax = vmax;
            cv::inRange(hsv, cv::Scalar(0, smin, MIN(_vmin,_vmax)),
                    cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

            // init meanshift
            if (fnumber == 0) {
                // calculate histogram
                histimg = cv::Mat::zeros(input.size(), CV_8UC3);

                cv::Mat myroi(hue, roi), maskroi(mask, roi);
                cv::calcHist(&myroi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                cv::normalize(hist, hist, 0, 255, CV_MINMAX);


                histimg = cv::Scalar::all(0);
                int binW = histimg.cols / hsize;
                cv::Mat buf(1, hsize, CV_8UC3);
                for( int i = 0; i < hsize; i++ )
                    buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180./hsize), 255, 255);
                cvtColor(buf, buf, CV_HSV2BGR);

                for( int i = 0; i < hsize; i++ )
                {
                    int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                    cv::rectangle( histimg, cv::Point(i*binW,histimg.rows),
                                   cv::Point((i+1)*binW,histimg.rows - val),
                                   cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8 );
                }
                // feature video size
                frameWidth = getFrameSize().width;
                frameHeight = getFrameSize().height;
                featureHeight = frameHeight / 2.0;
                featureWidth = roi.width * featureHeight / roi.height;
                featureSize = cv::Size(featureWidth, featureHeight);

                createTemp(cv::Size(frameWidth + featureWidth, frameHeight));
            }

            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            backproj &= mask;
            cv::meanShift(backproj, roi, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1 ));

            // kalman filtering
            if (isKalman) {
                if (fnumber == 0) {
                    KF.transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);

                    // init kalman filter
                    KF.statePre.at<float>(0) = roi.x;
                    KF.statePre.at<float>(1) = roi.y;
                    KF.statePre.at<float>(2) = 0;
                    KF.statePre.at<float>(3) = 0;

                    setIdentity(KF.measurementMatrix);
                    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
                    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
                    setIdentity(KF.errorCovPost, cv::Scalar::all(.1));

                    cv::Mat estimated = KF.correct(measurement);
                    cv::Point statePt(estimated.at<float>(0), estimated.at<float>(1));
                } else {
                    // kalman filtering roi position
                    cv::Mat prediction = KF.predict();
                    cv::Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

                    // Get roi position
                    measurement(0) = roi.x;
                    measurement(1) = roi.y;

                    // The "correct" phase
                    cv::Mat estimated = KF.correct(measurement);
                    cv::Point statePt(estimated.at<float>(0), estimated.at<float>(1));

                    roi.x = statePt.x;
                    roi.y = statePt.y;
                }
            }

            temp = input(roi).clone();
            cv::resize(temp, featureBefore, featureSize);
            featureBefores.push_back(featureBefore.clone());

            if (isGrabcut) {
                // calculate foreground mask
                foreMask = extractor->getForegroundMask(input, roi);
                foreMasks.push_back(foreMask);
            }

            rois.push_back(roi);
        }

        temp.convertTo(temp, CV_32FC3);
        windows.push_back(temp.clone());

        // spatial filtering
        std::vector<cv::Mat> pyramid;
        spatialFilter(temp, pyramid);
        downSampledFrames.push_back(pyramid.at(levels-1));

        // update process
        std::string msg= "Spatial Filtering...";
        emit updateProcessProgress(msg, floor((fnumber++) * 100.0 / length));
    }
    if (isStop()){
        emit closeProgressDialog();
        fnumber = 0;
        return;
    }
    emit closeProgressDialog();

    // concat all the frames into a single large Mat
    // where each column is a reshaped single frame
    // (for processing convenience)
    concat(downSampledFrames, videoMat);

    // temporal filtering
    temporalFilter(videoMat, filtered);

    // amplify color motion
    amplify(filtered, filtered);

    // de-concat the filtered image into filtered frames
    deConcat(filtered, downSampledFrames.at(0).size(), filteredFrames);

    // amplify each frame
    // by adding frame image and motions
    // and write into video
    fnumber = 0;
    for (int i=0; i<length-1 && !isStop(); ++i) {

        upsamplingFromGaussianPyramid(filteredFrames.at(i), levels, motion);

        cv::resize(motion, motion, windows.at(i).size());

        if (fnumber > 0)
            temp = windows.at(i) + motion;

        // convert back to ntsc color space
        output = temp.clone();
        double minVal, maxVal;
        minMaxLoc(output, &minVal, &maxVal); //find minimum and maximum intensities
        output.convertTo(output, CV_8UC3, 255.0/(maxVal - minVal),
                  -minVal * 255.0/(maxVal - minVal));

        // copy back the roi image to original frame
        if (isLabeled && isTrack) {
            cv::Rect currentRoi = rois.at(i);
            input = frames.at(i).clone();
            cv::Mat result = cv::Mat(input, currentRoi);
            output.copyTo(result);
            output = input;

            // draw the searched result if neccessary
            if (showROI)
                cv::rectangle(output, currentRoi, cv::Scalar(0,255,0));
        }

        // multi-resolution blending input and output
        if (isLabeled && isGrabcut) {
            cv::Mat foreground, background;
            frames.at(i).convertTo(background, CV_32F, 1.0/255.0f);
            output.convertTo(foreground, CV_32F, 1.0/255.0f);

            cv::Mat_<float> mask;
            foreMask = foreMasks.at(i);
            foreMask.convertTo(mask, CV_32F, 1.0/255.0f);

            output = LaplacianBlend(foreground, background, mask);
            output.convertTo(output, CV_8UC3, 255.0, 1.0/255.0);
        }

        // concat the feature image
        if (isLabeled && isTrack) {
            featureBefore = featureBefores.at(i);
            featureAfter = output(rois.at(i)).clone();
            cv::resize(featureAfter, featureAfter, featureSize);

            cv::Mat resultFrame(cv::Size(frameWidth + featureWidth, frameHeight), CV_8UC3, cv::Scalar(255, 255, 255));
            cv::Mat resultROI;
            resultROI = cv::Mat(resultFrame, cv::Rect(0, 0, frameWidth, frameHeight));
            output.copyTo(resultROI);
            resultROI = cv::Mat(resultFrame, cv::Rect(frameWidth, 0, featureWidth, featureHeight));
            featureBefore.copyTo(resultROI);
            resultROI = cv::Mat(resultFrame, cv::Rect(frameWidth, featureHeight, featureWidth, featureHeight));
            featureAfter.copyTo(resultROI);
            output = resultFrame.clone();
        }

        tempWriter.write(output);
        std::string msg= "Amplifying...";
        emit updateProcessProgress(msg, floor((fnumber++) * 100.0 / length));
    }
    if (!isStop()) {
        emit revert();
    }
    emit closeProgressDialog();

    // release the temp writer
    tempWriter.release();

    // change the video to the processed video
    setInput(tempFile);

    // jump back to the original position
    jumpTo(pos);
}


/** 
 * writeOutput	-	write the processed result
 *
 */
void VideoProcessor::writeOutput()
{
    cv::Mat input;

    // if no capture device has been set
    if (!isOpened() || !writer.isOpened())
        return;

    // save the current position
    long pos = curPos;
    
    // jump to the first frame
    jumpTo(0);

    while (getNextFrame(input)) {

        // write output sequence
        if (outputFile.length()!=0)
            writeNextFrame(input);
    }

    // set the modify flag to false
    modify = false;

    // release the writer
    writer.release();

    // jump back to the original position
    jumpTo(pos);
}

/** 
 * revertVideo	-	revert playing
 *
 */
void VideoProcessor::revertVideo()
{
    // pause the video
    jumpTo(0);    
    curPos = 0;
    pauseIt();
    emit updateProgressBar();
}

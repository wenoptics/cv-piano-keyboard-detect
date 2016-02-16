/// ??
/// Read a piano keyboard image,
///		draw the ROI in "src" window,
///		then the HoughLine result in "detected line" window
/// 
/// wenoptk 2016 Jan. 30

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

#define RND_COLOR Scalar(rand() & 255, rand() & 255, rand() & 255)

static void on_trackbar(int, void*);
void do_perspective_transform();

int key;

void onMousePersp(int event, int x, int y, int flags, void* param);
std::vector<cv::Point2f> VertexPersp; //perspective调整的源4个点
Mat onMousePerspImage;

Rect ROI;
Mat imgROI;

Mat src, src_gray;
Mat grad;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;

#pragma region ControlWindowParams
/// Param: Canny
int pCannyT1 = 50;
int pCannyT2 = 200;

/// Param: HoughLine
int pHoughT = 28;

/// Param: Adaptive Threshold
int pAdaptThres_blockSize = 26;
int pAdaptThres_C = 2;

/// Param: morph
int pMorph_size = 2;
#pragma endregion


void initWindow() {
	namedWindow("Control", CV_WINDOW_NORMAL);

	createTrackbar("pCannyT1", "Control", &pCannyT1, 1000, on_trackbar);
	createTrackbar("pCannyT2", "Control", &pCannyT2, 1000, on_trackbar);

	createTrackbar("pHoughT", "Control", &pHoughT, 100, on_trackbar);

	createTrackbar("pAdaptThres_blockSize", "Control", &pAdaptThres_blockSize, 300, on_trackbar);
	createTrackbar("pAdaptThres_C", "Control", &pAdaptThres_C, 500, on_trackbar);

	createTrackbar("pMorph_size", "Control", &pMorph_size, 20, on_trackbar);

	namedWindow("src - Draw Perspective(4 points)", 1);


#pragma region mousePointPerspective4Points
	setMouseCallback("src - Draw Perspective(4 points)", onMousePersp);
#pragma endregion

}

static void updateROI(){
	imgROI = src(ROI);
	on_trackbar(0, 0);
}

static void onMousePersp(int event, int x, int y, int flag, void* param) {
	static Scalar theColor(0, 0, 255);

	/// when mouse down
	if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN){}

	/// when mouse up
	if (event == EVENT_LBUTTONUP || event == EVENT_RBUTTONUP){

		printf("[onMousePersp] mouse up( %d, %d) ", x, y);
		printf("VertexPersp.size() ==  %d \n", VertexPersp.size());

		if (VertexPersp.size() < 4) {
			if (VertexPersp.size() == 0) {
				src.copyTo(onMousePerspImage);
			}
			cv::Point2f thePoint(x, y);
			VertexPersp.push_back(thePoint);

			cv::circle(onMousePerspImage, thePoint, 3, theColor, 3);
			if (VertexPersp.size() > 1) {
				// draw line to connect the latter one;
				cv::line(onMousePerspImage, thePoint, *(VertexPersp.end() - 2), theColor);
			}

			cv::putText(onMousePerspImage,
				std::to_string(VertexPersp.size()),
				thePoint,
				FONT_HERSHEY_SCRIPT_SIMPLEX, 1, theColor);

			cv::imshow("src - Draw Perspective(4 points)", onMousePerspImage);

			if (VertexPersp.size() == 4) {
				/// connect the first vertex
				cv::line(onMousePerspImage, thePoint, VertexPersp.front(), theColor);
				cv::imshow("src - Draw Perspective(4 points)", onMousePerspImage);

				// do perspective transform
				do_perspective_transform();
			}

		}
		else {
			VertexPersp.clear();

			// clear the canvas
			src.copyTo(onMousePerspImage);
			cv::imshow("src - Draw Perspective(4 points)", onMousePerspImage);
			printf("cleared \n");
		}


	}

	/// when drag
	if (flag == EVENT_FLAG_LBUTTON || flag == EVENT_FLAG_RBUTTON){

	}
}

int c;

void do_perspective_transform() {
	Mat matPTransform;

	// 0,0  0,h  w,h  w,0
	// w == 393 ; h == 135
	int w = 393;
	int h = 135;
	vector<Point2f> dst_transform
		= { Point2f(0, 0), Point2f(0, h), Point2f(w, h), Point2f(w, 0) };

	Mat imgAfterPerspTrans;

	matPTransform = cv::getPerspectiveTransform(&VertexPersp[0], &dst_transform[0]);

	cv::warpPerspective(src, imgAfterPerspTrans, matPTransform, Size(w, h));
	//imshow("AfterPerspTrans", imgAfterPerspTrans);

	/// show the image
	imgROI = imgAfterPerspTrans;
	on_trackbar(0, 0);
}

void do_back_perspective_transform(Mat &src_image, Mat& dst_image) {
	Mat matPTransform;

	// 0,0  0,h  w,h  w,0
	vector<Point2f> dst_transform
		= {
		Point2f(0, 0),
		Point2f(0, src_image.rows),
		Point2f(src_image.cols, src_image.rows),
		Point2f(src_image.cols, 0)
	};

	vector<Point2f> vec_4points = VertexPersp;

	if (vec_4points.size() == 0){
		vec_4points = dst_transform;
	}

	matPTransform = cv::getPerspectiveTransform(&dst_transform[0], &vec_4points[0]);
	cv::warpPerspective(src_image, dst_image, matPTransform, src.size());

}

static void on_trackbar(int, void*) {

	cv::cvtColor(imgROI, grad, CV_BGR2GRAY);
	
#if 1 /// use adative threshold
	cv::adaptiveThreshold(grad, grad, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
		pAdaptThres_blockSize * 2 + 1, pAdaptThres_C / 10);
	cv::imshow("adative threshold", grad);
#endif

	Mat element = getStructuringElement(MorphShapes::MORPH_RECT,
		Size(2 * pMorph_size + 1, 2 * pMorph_size + 1), Point(pMorph_size, pMorph_size));
#if 1 /// do MORPH op

	Mat imgMorph = Mat();

	/// do MORPH_Open	
	cv::morphologyEx(grad, grad, MORPH_OPEN, element);

	/// do MORPH_Close
	cv::morphologyEx(grad, grad, MORPH_CLOSE, element);
	imgMorph = grad.clone();
	cv::imshow("MORPH o - c", grad);
#endif
	
#if 1 /// contours
	//获取轮廓(不包括轮廓内的轮廓)
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(imgMorph.clone(), contours,
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	cv::Mat imgContourResult(grad.size(), CV_8U, cv::Scalar(255));

	cv::drawContours(imgContourResult, contours,
		-1, cv::Scalar(0), 3);

	Mat imgContoursPoly = imgROI.clone();
	Mat imgCoutoursRect = imgROI.clone();
	
	std::vector<double> contoursArea = vector<double>();

	// for each contours
	for (std::vector<cv::Point> c : contours) {

		// get the up-right rectangle
		Rect rect = cv::boundingRect(c);

		// draw the rectangle
		cv::rectangle(imgCoutoursRect, rect, RND_COLOR);

		// testing the approximate polygon  
		std::vector<cv::Point> poly;
		cv::approxPolyDP(cv::Mat(c), poly,
			7, // accuracy of the approximation  
			false); // closed shape 

		// calc the area
		double area = cv::contourArea(c);
		contoursArea.push_back(area);

		// get mass center 
		// compute all moments  
		cv::Moments mom = cv::moments(cv::Mat(c));
		// draw mass center  
		cv::circle(imgCoutoursRect,
			// position of mass center converted to integer  
			cv::Point(mom.m10 / mom.m00, mom.m01 / mom.m00),
			2, cv::Scalar(0), 2); // draw black dot  
		cv::putText(imgCoutoursRect, 
			std::to_string((int)area),
			cv::Point(mom.m10 / mom.m00, mom.m01 / mom.m00),
			FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, RND_COLOR);

		//debug print
		printf("%f,\n", area);

		// draw the poly
		cv::polylines(imgContoursPoly, poly, false, RND_COLOR);
	}
	
	cv::imshow("drawContours", imgContourResult);
	cv::imshow("Contours Approx Poly", imgContoursPoly);
	cv::imshow("Contours rect", imgCoutoursRect);


	/// do area recognition
	Mat imgRecoResult = imgROI.clone();
	int i = 0;
	double maxArea = *std::max_element(contoursArea.begin(), contoursArea.end());
	double areaThreshold = maxArea * 0.75;
	printf("!the max is %f,\n", maxArea);

	for (std::vector<cv::Point> c : contours) {
		// get the up-right rectangle
		Rect rect = cv::boundingRect(c);

		if (contoursArea[i++] > areaThreshold) {
			// draw the rectangle
			cv::rectangle(imgRecoResult, rect, Scalar(0, 255, 0, 0.5), CV_FILLED);
		}
		else{
			// draw the rectangle
			cv::rectangle(imgRecoResult, rect, Scalar(0, 0, 255, 0.5), CV_FILLED);
		}
	}

	cv::imshow("Area Recognition Result", imgRecoResult);

#endif

	Mat cdst;
	// overlay on the source image
	imgRecoResult.copyTo(cdst);
	
#if 1 /// project the img back to the original perspetation
	Mat imgBackTrans;
	do_back_perspective_transform(cdst, imgBackTrans);
	imshow("back transform", imgBackTrans);
#endif

}


/** @function main */
int main(int argc, char** argv)
{

	/// Load an image
	src = imread("D:/WorkSpace/WIN/OpenCV/prog/cv_piano_keyboard/test_photo/3_s.png");

	if (!src.data)
	{
		return -1;
	}

	imshow("src - Draw Perspective(4 points)", src);

	src.copyTo(imgROI);

	initWindow();
	on_trackbar(0, 0);

	while (true){
		key = cvWaitKey(0);
		if (key == 27){
			break;
		}
	}

	return 0;
}
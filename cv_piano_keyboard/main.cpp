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

bool useSobel = false;

void onMouse(int event, int x, int y, int flags, void* param);

Point VertexOne, VertexThree;//長方形的左上點和右下點
Scalar Color;//框框顏色
int Thickness;//框框粗細
int Shift;//框框大小(0為正常)
int key;//按鍵碼

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
int pMorph_size = 4;
#pragma endregion


void initWindow() {
	namedWindow("Control", CV_WINDOW_NORMAL);

	createTrackbar("pCannyT1", "Control", &pCannyT1, 1000, on_trackbar);
	createTrackbar("pCannyT2", "Control", &pCannyT2, 1000, on_trackbar);

	createTrackbar("pHoughT", "Control", &pHoughT, 100, on_trackbar);

	createTrackbar("pAdaptThres_blockSize", "Control", &pAdaptThres_blockSize, 300, on_trackbar);
	createTrackbar("pAdaptThres_C", "Control", &pAdaptThres_C, 500, on_trackbar);

	createTrackbar("pMorph_size", "Control", &pMorph_size, 20, on_trackbar);

	namedWindow("sobel", CV_WINDOW_AUTOSIZE);

	namedWindow("src", 1);
	namedWindow("src-perspective(4 points)", 1);

#pragma region mouseDragRectSetup
	VertexOne = cvPoint(0, 0);
	VertexThree = cvPoint(0, 0);
	Color = CV_RGB(0, 255, 0);
	Thickness = 2;
	Shift = 0;
	key = 0;
	setMouseCallback("src", onMouse);
#pragma endregion

#pragma region mousePointPerspective4Points
	setMouseCallback("src-perspective(4 points)", onMousePersp);
#pragma endregion

}

static void updateROI(){
	imgROI = src(ROI);
	on_trackbar(0, 0);
}

static void onMousePersp(int event, int x, int y, int flag, void* param) {
	static Scalar theColor(0, 0, 255);

	/// when mouse down
	if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN){		}

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
				cv::line(onMousePerspImage, thePoint, *(VertexPersp.end()-2), theColor);
			}

			cv::putText(onMousePerspImage,
				std::to_string(VertexPersp.size()), 
				thePoint, 
				FONT_HERSHEY_SCRIPT_SIMPLEX, 1, theColor);

			imshow("src-perspective(4 points)", onMousePerspImage);

			if (VertexPersp.size() == 4) {
				/// connect the first vertex
				cv::line(onMousePerspImage, thePoint, VertexPersp.front(), theColor);
				imshow("src-perspective(4 points)", onMousePerspImage);
				
				// do perspective transform
				do_perspective_transform();
			}

		}
		else {
			VertexPersp.clear();

			// clear the canvas
			src.copyTo(onMousePerspImage);
			imshow("src-perspective(4 points)", onMousePerspImage);
			printf("cleared \n");
		}


	}

	/// when drag
	if (flag == EVENT_FLAG_LBUTTON || flag == EVENT_FLAG_RBUTTON){

	}
}

static void onMouse(int event, int x, int y, int flag, void* param){
	printf("( %d, %d) ", x, y);
	printf("The Event is : %d ", event);
	printf("The flags is : %d ", flag);
	printf("The param is : %d\n", param);

	Mat Image;

	/// left-upper co, when mouse down
	if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN){ 
		VertexOne = Point(x, y);
		//src.copyTo(Image);
	}
	/// right-down co, when mouse up
	if (event == EVENT_LBUTTONUP || event == EVENT_RBUTTONUP){
		VertexThree = Point(x, y);
		src.copyTo(Image);
		rectangle(Image, VertexOne, VertexThree, Color, Thickness, CV_AA, Shift);
		imshow("src", Image);
		ROI = Rect(VertexOne, VertexThree);

		//updateROI
		updateROI();

	}

	/// when drag
	if (flag == EVENT_FLAG_LBUTTON || flag == EVENT_FLAG_RBUTTON){
		VertexThree = Point(x, y);
		src.copyTo(Image);
		rectangle(Image, VertexOne, VertexThree, Color, Thickness, CV_AA, Shift);
		imshow("src", Image);
	}

	printf("VertexOne( %d, %d) ", VertexOne.x, VertexOne.y);
	printf("VertexThree( %d, %d)\n", VertexThree.x, VertexThree.y);
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

	matPTransform = cv::getPerspectiveTransform(&VertexPersp[0], &dst_transform[0]);

	Mat imgAfterPerspTrans;
	cv::warpPerspective(src, imgAfterPerspTrans, matPTransform, Size(w, h));
	//imshow("AfterPerspTrans", imgAfterPerspTrans);

	/// show the image
	imgROI = imgAfterPerspTrans;
	on_trackbar(0, 0);

}

static void on_trackbar(int, void*) {

#if 0 //useSobel

		GaussianBlur(imgROI, imgROI, Size(3, 3), 0, 0, BORDER_DEFAULT);

		/// Convert it to gray
		cvtColor(imgROI, src_gray, CV_BGR2GRAY);

		/// Generate grad_x and grad_y
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		imshow("sobel", grad);
	
#else
		cvtColor(imgROI, grad, CV_BGR2GRAY);
#endif

#if 0 /// do Gussasain blur
	cv::GaussianBlur(grad, grad, Size(3, 3),0);
	imshow("GaussianBlur", grad);
#endif

#if 1 /// use adative threshold
	cv::adaptiveThreshold(grad, grad, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 
		pAdaptThres_blockSize*2+1, pAdaptThres_C/10);
	imshow("adative threshold", grad);
#endif

	Mat element = getStructuringElement(MorphShapes::MORPH_RECT,
		Size(2 * pMorph_size + 1, 2 * pMorph_size + 1), Point(pMorph_size, pMorph_size));
#if 1 /// do MORPH op

	/// do MORPH_Open	
	cv::morphologyEx(grad, grad, MORPH_OPEN, element);

	/// do MORPH_Close
	cv::morphologyEx(grad, grad, MORPH_CLOSE, element);
	imshow("MORPH o - c", grad);
#endif

	Canny(grad, grad, pCannyT1, pCannyT2, 3);
	imshow("Canny", grad);


#if 1 /// contours
	//获取轮廓不包括轮廓内的轮廓  
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(grad.clone(), contours,
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cv::Mat imgContourResult(grad.size(), CV_8U, cv::Scalar(255));

	cv::drawContours(imgContourResult , contours ,  
		-1 , cv::Scalar(0) , 2) ;  

	Mat imgContoursPoly = imgROI.clone();

	// testing the approximate polygon  	
	for (std::vector<cv::Point> c : contours) {
		std::vector<cv::Point> poly;
		cv::approxPolyDP(cv::Mat(c), poly,
			5, // accuracy of the approximation  
			false); // closed shape 
		// draw the poly
		cv::polylines(imgContoursPoly, poly, false, RND_COLOR);
	} 

	cv::imshow("drawContours" , imgContourResult) ; 
	cv::imshow("Contours poly" , imgContoursPoly) ; 
#endif


	Mat cdst;
	// overlay on the source image
	imgROI.copyTo(cdst);

#if 1 /// use contours image to find lines
	imgContourResult.copyTo(grad);
	grad = -grad + 255; // invert it
#endif
#if 1 /// use HoughLines to find lines	
	vector<Vec2f> lines;
	// detect lines
	HoughLines(grad, lines, 1, CV_PI / 180, pHoughT );

	// draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];

		// limit the line in an angle range
		if (theta<CV_PI / 180 * 70 || theta>CV_PI / 180 * 110)
		{
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(cdst, pt1, pt2, RND_COLOR, 1);

		}
	}
#else /// or use HoughLineP to find lines
	vector<Vec4i> lines;
	cv::HoughLinesP(grad, lines, 1, CV_PI / 180, pHoughT, 30, 3);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(cdst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), RND_COLOR, 1, 8);
	}
	
#endif
	imshow("detected lines", cdst);

}


/** @function main */
int main(int argc, char** argv)
{

	/// Load an image
	src = imread("D:/WorkSpace/WIN/OpenCV/prog/opencv_test1/piano-raw/2_s.png");

	if (!src.data)
	{
		return -1;
	}

	imshow("src", src);
	imshow("src-perspective(4 points)", src);

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
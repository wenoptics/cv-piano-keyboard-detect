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

static void on_trackbar(int, void*);

bool useSobel = false;

void onMouse(int event, int x, int y, int flags, void* param);

Point VertexOne, VertexThree;//長方形的左上點和右下點
Scalar Color;//框框顏色
int Thickness;//框框粗細
int Shift;//框框大小(0為正常)
int key;//按鍵碼

void onMousePersp(int event, int x, int y, int flags, void* param);
std::vector<cv::Point> VertexPersp; //perspective调整的源4个点
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
int pHoughT = 150;
#pragma endregion


void initWindow() {
	namedWindow("Control", CV_WINDOW_NORMAL);

	createTrackbar("pCannyT1", "Control", &pCannyT1, 1000, on_trackbar);
	createTrackbar("pCannyT2", "Control", &pCannyT2, 1000, on_trackbar);

	createTrackbar("pHoughT", "Control", &pHoughT, 300, on_trackbar);

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
			cv::Point thePoint(x, y);
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
				// do something
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

static void on_trackbar(int, void*) {

	if (useSobel) {

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
	}
	else{
		cvtColor(imgROI, grad, CV_BGR2GRAY);
	}

#if 1 /// do Gussasain blur
	cv::GaussianBlur(grad, grad, Size(3, 3),0);
	imshow("GaussianBlur", grad);
#endif

	Canny(grad, grad, pCannyT1, pCannyT2, 3);
	imshow("Canny", grad);

	Mat cdst;

	// overlay on the source image
	imgROI.copyTo(cdst);

#if 0 /// use HoughLines to find lines	
	vector<Vec2f> lines;
	// detect lines
	HoughLines(grad, lines, 1, CV_PI / 180, pHoughT, 0, 0);

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
			line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);

		}
	}
#else /// or use HoughLineP to find lines
	vector<Vec4i> lines;
	cv::HoughLinesP(grad, lines, 1, CV_PI / 180, pHoughT, 30, 3);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(cdst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
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
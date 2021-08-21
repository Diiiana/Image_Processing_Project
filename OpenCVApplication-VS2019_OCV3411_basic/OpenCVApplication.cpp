// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "objdetect.hpp"

CascadeClassifier face_cascade; // cascade clasifier object for face
CascadeClassifier eyes_cascade; // cascade clasifier object for eyes

// red saturation result for the red reducing algorithm
Mat redSat;
// value of red selected by the user
int val = 0;

// converting RGB image to GRAY image
void imageToGrayScale() {
	
	// getting input image as color image
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;	// extracting its width and height for iteration
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);		// result is a one channel image

		for (int i = 0; i < height; i++) {		// iterating image to calculate the gray image
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src.at<Vec3b>(i, j);			// for each pixel from the color image
				float value = (pixel[2]*0.2126 + pixel[1]*0.7152 + pixel[0]*0.0722);	// we apply Luma formula
				dst.at<uchar>(i, j) = value;				// and update the result
			}
		}

		imshow("Input image is: ", src);	// input and output images displayed
		imshow("Output image is: ", dst);
		waitKey(0);
	}
}

// maximum of 3 float values, needed for RGB2HSV conversion
float maximum(float a, float b, float c) {
	return max(a, max(b, c));
}

// minimum of 3 float values, needed for RGB2HSV conversion
float minimum(float a, float b, float c) {
	return min(a, min(b, c));
}

// colored image to HSV - result is returned in given parameters
void RGB2HSV(Mat src, Mat* dstH, Mat* dstS, Mat* dstV) {

	float M, m, C, V, S, H;		// values of the minimum, maximum, hue, saturation, value needed for the algorithm

	int height = src.rows;		// extracting width and height
	int width = src.cols;

	*dstH = Mat(height, width, CV_8UC1);		// hue image is an image with only one channel
	*dstS = Mat(height, width, CV_8UC1);		// saturation image is an image with only one channel
	*dstV = Mat(height, width, CV_8UC1);        // value image is an image with only one channel

	for (int i = 0; i < height; i++)			// for each pixel in the input image
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b v3 = src.at<Vec3b>(i, j);
			float b = (float)v3[0] / 255;			// we modify its RGB values to scale the image
			float g = (float)v3[1] / 255;
			float r = (float)v3[2] / 255;

			M = maximum(r, g, b);				// select minimum and maximum from the resulting scaled values
			m = minimum(r, g, b);

			C = M - m;							// select their difference that would be applied in the image
			V = M;

			if (V != 0) {					  // if we can split by value 
				S = C / V;					  // saturation is the raport between difference and value
			}
			else {	
				S = 0;						 // else saturation is 0
			}

			if (C != 0) {					// if average is not 0
				if (M == r) {
					H = 60 * (g - b) / C;		// we have different ranges for H
				}
				if (M == g) {
					H = 120 + 60 * (b - r) / C;		// we fit H in its range
				}
				if (M == b) {
					H = 240 + 60 * (r - g) / C;
				}
			}
			else {
				H = 0;			// grayscale image takes black
			}

			if (H < 0) {	   // grayscale image takes maximum
				H += 360;
			}

			float H_norm = H * 255 / 360;			// as for the HSV is a circle representation and H is between 0..260, we need to resize its range
			float S_norm = S * 255;					// saturation and value are between 0..1, we need to scale them to 0-255 
			float V_norm = V * 255;

			dstH->at<uchar>(i, j) = H_norm;			// we update matrices with the resulting values
			dstS->at<uchar>(i, j) = S_norm;
			dstV->at<uchar>(i, j) = V_norm;
		}
	}
}

// method that modifies each eye color
Mat process(Mat frame, Point center, int radius, int eye) {
	Mat hsv;		// HSV image for input frame
	Mat mask;		// final mask
	Mat mask1;		// mask for [0-8] red
	Mat mask2;     // mask for [165-180] red

	int startRows = center.y - radius;		// we traverse the eye from center adjusted to radius
	int endRows = center.y + radius;
	int startCols = center.x - radius;
	int endCols = center.x + radius;

	int diametru = 0, start = 0, raza = 0;		// needed for pupil diameter

	cvtColor(frame, hsv, COLOR_BGR2HSV);  // HSV image for input frame

	inRange(hsv, Scalar(0, 70, 40), Scalar(8, 255, 255), mask1);       // mask for [0-8] red
	inRange(hsv, Scalar(165, 70, 40), Scalar(180, 255, 255), mask2);  // mask for [165-180] red

	mask = (mask1 | mask2);    // the final mask is their logic OR -> red can be either in [0-8] or in [165-180]
	if(eye == 0)			   // we display the final mask
		imshow("Mask", mask);

	// we take the line of pixels from the center of the eye and search for the pupil
	for (int iy = startRows + radius; iy < (startRows + (radius)+1) && !start; iy++) {
		for (int ix = startCols + 5; ix < endCols; ix++) {
			if (mask.at<uchar>(iy, ix) == 255) {	// when finding a white value in mask, we might find pupil
				start = 1;							// we mark pupil start
				diametru++;							// increment diameter of the pupil
			}
			else {
				if (start) {						// if we already started, and it's not a white value
					if (diametru < 6) {				// either it's not the pupil
						diametru = 0;				// and reset
					}
					else {
						if (diametru < 8) {			// or its a valid diameter, but too small, so we double it to iterate the full pupil
							diametru *= 2;
						}
						else {						// if it's a valid diameter, we stop
							break;
						}
					}
				}
			}
		}
	}

	if (eye == 1) {		// if eye is one, then it's the second one and needs smaller radius, being further than the face width
		raza = diametru / 2 + sqrt(diametru / 2);	// a verified formula to take only the pixels from the circle
	}
	else {
		raza = diametru / 2 + 2 * sqrt(diametru / 2);
	}

	// new center from where we start and iterate the eye in circle
	int centerX = startRows + radius;
	int centerY = startCols + radius;
	for (int v = centerX - raza; v < centerX + raza; v++) {
		// going left and right the circle from center
		for (int o = centerY; pow((o - centerY), 2) + pow((v - centerX), 2) <= pow(raza, 2); o--) {	
			if (mask.at<uchar>(v, o) == 255) {			// where mask is white (we have red), and the index is in mask's range of the eye
				if (v > center.y - radius / 2 && v < center.y + radius / 2 - 1 && o >= center.x - radius / 2 && o < center.x + radius / 2) {
					int red = (frame.at<Vec3b>(v, o)[0] + frame.at<Vec3b>(v, o)[1]) / 2;	// we take it's colors
					frame.at<Vec3b>(v, o)[2] = red;											// and modify by scaling each pixel to average of green and blue
					frame.at<Vec3b>(v, o)[1] = red;
					frame.at<Vec3b>(v, o)[0] = red;
					mask.at<uchar>(v, o) == 0;				// modify mask - needed for the area accuracy check (red pixels left)
				}
			}
		}
		// going left and right the circle from center
		for (int p = centerY + 1; pow((p - centerY), 2) + pow((v - centerX), 2) <= pow(raza, 2); p++) {
			if (mask.at<uchar>(v, p) == 255) {
				if (v > center.y - radius / 2 && v < center.y + radius / 2 - 1 && p >= center.x - radius / 2 && p < center.x + radius / 2) {
					int red = (frame.at<Vec3b>(v, p)[0] + frame.at<Vec3b>(v, p)[1]) / 2; // we take it's colors
					frame.at<Vec3b>(v, p)[2] = red;						// and modify by scaling each pixel to average of green and blue
					frame.at<Vec3b>(v, p)[1] = red; 
					frame.at<Vec3b>(v, p)[0] = red;             
					mask.at<uchar>(v, p) == 0;               // modify mask - needed for the area accuracy check (red pixels left)
				}
			}
		}
	}

	cvtColor(frame, hsv, COLOR_BGR2HSV);			// convert the new image to create a new mask
	inRange(hsv, Scalar(0, 70, 40), Scalar(8, 255, 255), mask1);
	inRange(hsv, Scalar(165, 70, 40), Scalar(180, 255, 255), mask2);
	mask = (mask1 | mask2);			// the new mask is used for accuracy

	// we get the initial area
	int area = (center.y + radius / 2 - 3 - (center.y - radius / 2 + 3)) * (center.x + radius / 2 + 3 - (center.x - radius / 2 - 1));

	int count = 0;	// for counting red pixels left
	// we iterate the eye again
	for (int v = center.y - radius / 2 + 3; v < center.y + radius / 2 - 3; v++) {
		for (int o = center.x - radius / 2 - 1; o < center.x + radius / 2 + 3; o++) {
			if (mask.at<uchar>(v, o) == 255) {	// where new mask is white, we increment inaccuracy
				count++;
			}
		}
	}

	// if inaccuracy is greated than 1/5 eye, we traverse again with a bigger radius
	if (area / (float)count <= 5) {
		for (int v = center.y - radius / 2 + 3; v < center.y + radius / 2 - 2; v++) {
			for (int o = center.x - radius / 2; o <= center.x + radius / 2; o++) {
				if (mask.at<uchar>(v, o) == 255) {
					int red = (frame.at<Vec3b>(v, o)[0] + frame.at<Vec3b>(v, o)[1]) / 2;
					frame.at<Vec3b>(v, o)[2] = red;	// modify value
					mask.at<uchar>(v, o) == 0;		// setting mask
				}
			}
		}
	}

	return frame;
}

// function for detecting face and eyes
void detectFaceAndEyes(const string& window_name, Mat frame, int minFaceSize, int minEyeSize)
{
	// input image conversion to 
	Mat inFrame;
	cvtColor(frame, inFrame, COLOR_RGB2GRAY);

	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	Mat eyeDetection = frame.clone();

	// minFaceSize - minimum size of the ROI in which a Face is searched
	// minEyeSize - minimum size of the ROI in which an Eye is searched
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
			Size(minEyeSize, minEyeSize));

		
		for (int j = 0; j < eyes.size(); j++)
		{
			// get the center of the eye
			// relative to left corner of the image
			Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
				faces[i].y + eyes[j].y + eyes[j].height * 0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height) * 0.2);
			circle(eyeDetection, center, radius, Scalar(255, 0, 0), 1, 8, 0);
			frame = process(frame, center, radius, j);	// process each eye
		}
	}
	imshow("Eyes", eyeDetection);	// display eye detection and frame
	imshow("Result", frame);
}

/* void eliminareRosu() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3);
		Mat hsv;
		cvtColor(src, hsv, COLOR_BGR2HSV);

		int val = -9999;
		while (val == -9999) {
			printf("Valoarea saturarii: ");
			scanf("%d", &val);
			printf("\n");
		}

		Mat h = Mat(height, width, CV_8UC3);
		Mat s = Mat(height, width, CV_8UC3);
		Mat v = Mat(height, width, CV_8UC3);

		Mat h1 = Mat(height, width, CV_8UC1);
		Mat s1 = Mat(height, width, CV_8UC1);
		Mat v1 = Mat(height, width, CV_8UC1);

		vector<Mat> sl;
		sl.push_back(h1);
		sl.push_back(s1);
		sl.push_back(v1);
		split(hsv, sl);

		RGB2HSV(src, &h, &s, &v);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				s1.at<uchar>(i, j) = val;

			}
		}
		Mat result = Mat(height, width, CV_8UC3);
		Mat gr = Mat(height, width, CV_8UC1);
		Mat gr1 = Mat(height, width, CV_8UC3);

		vector<Mat> matrices;
		matrices.push_back(h1);
		matrices.push_back(s1);
		matrices.push_back(v1);
		cv::merge(matrices, gr);

		cvtColor(gr, result, COLOR_HSV2BGR);

		imshow("Input image is: ", src);
		imshow("Output image is: ", result);
		waitKey(0);
	}
}
*/
// function for interactive trackbar
static void on_trackbar(int, void*)
{
	int height = redSat.rows;
	int width = redSat.cols;

	Mat dst = redSat.clone();		// result will be modified in red channel with input global value
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<Vec3b>(i, j)[2] = val;
		}
	}

	imshow("Output", dst);			// display the result
}

void redRemoval() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {			// reading input color image
		redSat = imread(fname, CV_LOAD_IMAGE_COLOR);

		imshow("image", redSat);		// display the final result
		namedWindow("image");		    // associate trackbar to image
		createTrackbar("Saturation", "image", &val, 255, on_trackbar);	// with function, minimum and maximum value
		waitKey(0);
	}
}

// eye detection for the red eye removal
void detectEyes() {
	char fname[MAX_PATH];

	// Load the cascades
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

	// if load failed, we print erroe message
	if (!face_cascade.load(face_cascade_name)) {
		printf("Error loading face cascades !\n");
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("Error loading eyes cascades !\n");
		return;
	}

	// we call processing function for the input image with its features
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dst = src.clone();

		dst = src.clone();

		int minFaceSize = 30;
		int minEyeSize = minFaceSize / 5; // according to the anthropomorphic properties of the face

		detectFaceAndEyes("Result ", dst, minFaceSize, minEyeSize);

		imshow("Input image is: ", src);
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Grayscale\n");
		printf(" 2 - Red eyes removal\n");
		printf(" 3 - Red removal\n");

		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			imageToGrayScale();
			break;
		case 2:
			detectEyes();
			break;
		case 3:
			redRemoval();
			break;
		}
	} while (op != 0);
	return 0;
}
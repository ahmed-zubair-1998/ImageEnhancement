#include <iostream>
#include <opencv2/opencv.hpp>

#include "contrast_enhancement.h"
#include "agarwal.h"
#include "agcwd.h"

using namespace cv;
using namespace std;

void strech(const Mat &img, Mat &grey) {
	/*
	Get luminance channel of image

	Parameters:
		img: Input image
		grey: output image
	*/

	Mat chan[3];
	Mat temp;
	split(img, chan);
	chan[0] *= 0.299;
	chan[1] *= 0.587;
	chan[2] *= 0.114;
	grey = chan[0] + chan[1] + chan[2];
}

void hue_correction(Mat &prev, const Mat &img, const Mat &grey, Mat &enhanced) {
	/*
	Performs hue correction.

	Paraeters:
		prev: Input enhanced image
		img: Original image
		grey: luminance channel of original image
		enhanced: output enhanced image qith hue correctionn
	*/

	vector<Mat> chan(3);
	split(img, chan);
	Mat temp, temp2;
	divide(265 - prev, 265 - grey, temp);
	for (int i = 0; i < 3; i++) {
		multiply(chan[i] - grey, temp, chan[i]);
		chan[i] += prev;
	}
	temp.release();
	Mat less;
	merge(chan, less);
	split(img, chan);
	divide(prev, grey, prev);
	double max_v;
	minMaxIdx(prev, nullptr, &max_v, nullptr, nullptr);
	for (int i = 0; i < 3; i++) {
		multiply(chan[i], prev, chan[i]);
	}
	Mat great;
	merge(chan, great);

	threshold(prev, prev, 1, 1, 0);
	for (int i = 0; i < 3; i++) {
		chan[i] = prev.clone();
	}
	merge(chan, prev);
	
	multiply(Scalar(1,1,1) - prev, less, less);
	multiply(prev, great, great);
	enhanced = less + great;
}

void global_contrast(const Mat &grey, Mat &global) {
	/*
	Global contrast enhancement. For JHE, see agarwal.h

	Parameters:
		grey: Luminance channel of original image
		global: global contrast enhanced output image
	*/

	grey.convertTo(global, CV_8UC1);
	JHE(global, global);
	global.convertTo(global, CV_32FC1);
}

void local_contrast(const Mat &grey, Mat &local) {
	/*
	Local  contrast enhancement. CLAHE is used.

	Parameters:
		grey: Luminance channel of original image
		local: local contrast enhanced output image
	*/

	grey.convertTo(local, CV_8UC1);
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(2);
	clahe->apply(local, local);
	local.convertTo(local, CV_32FC1);
}

void contrastEnhancement(const cv::Mat& src, cv::Mat& dst)
{
	/*
	Contrast Enhancement.

	Parameters:
		src: Input image
		dst: contrast enhanced output
	*/

	Mat img;
	normalize(src, img, 0, 255, NORM_MINMAX, CV_32FC3);
	Mat grey;
	strech(img, grey);	
	Mat global, local;
	Mat g_enhanced, l_enhanced;
	global_contrast(grey, global);
	hue_correction(global, img, grey, g_enhanced);
	global.release();
	local_contrast(grey, local);
	hue_correction(local, img, grey, l_enhanced);
	local.release();
	grey.release();
	img.release();
	
	/*imshow("local", l_enhanced/255);
	imshow("global", g_enhanced/255);*/

	//Weight Maps Calculation
	Mat lg, gg;
	strech(l_enhanced, lg);
	strech(g_enhanced, gg);

	Mat lpl, lpg, bl, bg;
	Laplacian(lg, lpl, CV_32FC1, 3, 1, 0, BORDER_DEFAULT);
	Laplacian(gg, lpg, CV_32FC1, 3, 1, 0, BORDER_DEFAULT);
	
	//1.99s
	Mat temp = lg - 0.5;
	multiply(temp, temp, temp);
	bl = temp / 0.08;
	temp = gg - 0.5;
	multiply(temp, temp, temp);
	bg = temp / 0.08;

	//1.61s
	Mat wdl, wdg;
	cv::min(bl, lpl, wdl);
	cv::min(bg, lpg, wdg);

	normalize(wdl, wdl, 255, 0, NORM_MINMAX);
	normalize(wdg, wdg, 255, 0, NORM_MINMAX);

	lpl.release(); lpg.release(); bl.release(); bg.release();

	Mat wl, wg;
	divide(wdl, wdl + wdg, wl);
	divide(wdg, wdl + wdg, wg);

	wdl.release(); wdg.release();
	
	vector<Mat> chan(3);
	for (int i = 0; i < 3; i++) {
		chan[i] = wl.clone();
	}
	merge(chan, wl);
	for (int i = 0; i < 3; i++) {
		chan[i] = wg.clone();
	}
	merge(chan, wg);
	for (int i = 0; i < 3; i++) {
		chan[i].release();
	}

	//Weight and Images Blending using Image Pyramids
	vector<Mat> gaussianA, gaussianB, lapA, lapB, lap;

	int pyrScale = 1;

	lg = l_enhanced; gg = g_enhanced;

	for (int i = 0; i < pyrScale; i++) {
		gaussianA.push_back(lg);
		pyrDown(lg, lg);
	}

	for (int i = 0; i < pyrScale; i++) {
		gaussianB.push_back(gg);
		pyrDown(gg, gg);
	}

	lapA.push_back(gaussianA[pyrScale - 1]);
	for (int i = pyrScale - 1; i > 0; i--) {
		pyrUp(gaussianA[i], lg);
		lapA.push_back(gaussianA[i - 1] - lg);
	}

	lapB.push_back(gaussianB[pyrScale - 1]);
	for (int i = pyrScale - 1; i > 0; i--) {
		pyrUp(gaussianB[i], gg);
		lapB.push_back(gaussianB[i - 1] - gg);
	}

	for (int i = pyrScale - 1; i >= 0; i--) {
		gaussianA[i] = wl.clone();
		pyrDown(wl, wl);
	}

	for (int i = pyrScale - 1; i >= 0; i--) {
		gaussianB[i] = wg.clone();
		pyrDown(wg, wg);
	}

	for (int i = 0; i < pyrScale; i++) {
		multiply(gaussianA[i], lapA[i], lg);
		multiply(gaussianB[i], lapB[i], gg);
		lap.push_back(lg + gg);
	}

	dst = lap[0];
	for (int i = 1; i < pyrScale; i++) {
		pyrUp(dst, dst);
		add(dst, lap[i], dst);
	}
	
	dst.convertTo(dst, CV_8UC3);
}
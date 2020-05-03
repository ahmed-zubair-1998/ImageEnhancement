/*
	Code for "Guided Image Filtering". 

	Reference:
		K. He, J. Sun and X. Tang, "Guided Image Filtering," 
		in IEEE Transactions on Pattern Analysis and Machine Intelligence, 
		vol. 35, no. 6, pp. 1397-1409, June 2013

	Code Obtained From:
		https://github.com/atilimcetin/guided-filter
*/

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>

class GuidedFilterImpl;

class GuidedFilter
{
public:
	GuidedFilter(const cv::Mat &I, int r, double eps);
	~GuidedFilter();

	cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
	GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth = -1);

#endif
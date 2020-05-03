/*
	This contrast enhancement module is inspired by "Naturalness preservation image contrast enhancement via histogram modification".
	Majority of the work is implementation of this research paper.

	Reference:
		Tian, Q.C. and Cohen, L.D., 2018, April. Naturalness preservation image contrast enhancement via histogram modification. 
		In Ninth International Conference on Graphic and Image Processing (ICGIP 2017) 
		(Vol. 10615, p. 106152U). International Society for Optics and Photonics.
		
*/

void contrastEnhancement(const cv::Mat& src, cv::Mat& dst);
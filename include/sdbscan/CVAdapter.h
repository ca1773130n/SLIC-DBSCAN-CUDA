#pragma once
#include "VColor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace sdbscan {
	enum ColorFormat ConvertFromCVColorFormat(int cvFormat) {
		switch (cvFormat) {
			case CV_8UC3:
				return COLOR_RGB_888;
		}
		return COLOR_INVALID;
	}

	int ConvertToCVColorFormat(enum ColorFormat format) {
		switch (format) {
			case COLOR_RGB_888:
				return CV_8UC3;
			case COLOR_RGBA_8888:
				return CV_8UC4;
			case COLOR_RGB_FLOAT:
				return CV_32FC3;
			case COLOR_RGBA_FLOAT:
				return CV_32FC4;
			case COLOR_YUV_420:
				return CV_8UC1;
		}
		return 0;
	}

	void UnpackCVMat(cv::Mat& input, unsigned char *output) {
		const unsigned w = input.cols;
		const unsigned h = input.rows;
		const unsigned size = w * h;
		const unsigned numChannels = input.channels();
		int idx = -1;

		for (size_t i = 0; i < h; ++i) {
			for (size_t j = 0; j < w; ++j) {
				cv::Vec3b ip = input.at<cv::Vec3b>(i, j);
				idx = numChannels * (i * w + j);
				for (size_t c = 0; c < numChannels; ++c) {
					output[idx + c] = ip[c];
				}
			}
		}
	}

	void ConvertMatToUC4ImgRGB(const cv::Mat& inimg, UChar4Image* outimg)
	{
		Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

		for (int y = 0; y < outimg->noDims.y; y++)
			for (int x = 0; x < outimg->noDims.x; x++)
			{
				int idx = x + y * outimg->noDims.x;
				outimg_ptr[idx].b = inimg.at<cv::Vec3b>(y, x)[0];
				outimg_ptr[idx].g = inimg.at<cv::Vec3b>(y, x)[1];
				outimg_ptr[idx].r = inimg.at<cv::Vec3b>(y, x)[2];
				outimg_ptr[idx].a = 255;
			}
	}

	void ConvertUC4ImgToMatRGB(const UChar4Image* inimg, cv::Mat& outimg)
	{
		const Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

		for (int y = 0; y < inimg->noDims.y; y++)
			for (int x = 0; x < inimg->noDims.x; x++)
			{
				int idx = x + y * inimg->noDims.x;
				outimg.at<cv::Vec3b>(y, x)[0] = inimg_ptr[idx].b;
				outimg.at<cv::Vec3b>(y, x)[1] = inimg_ptr[idx].g;
				outimg.at<cv::Vec3b>(y, x)[2] = inimg_ptr[idx].r;
			}
	}

	void ConvertUC4ImgToMatRGBA(const UChar4Image* inimg, cv::Mat& outimg)
	{
		const Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

		for (int y = 0; y < inimg->noDims.y; y++)
			for (int x = 0; x < inimg->noDims.x; x++)
			{
				int idx = x + y * inimg->noDims.x;
				outimg.at<cv::Vec4b>(y, x)[0] = inimg_ptr[idx].b;
				outimg.at<cv::Vec4b>(y, x)[1] = inimg_ptr[idx].g;
				outimg.at<cv::Vec4b>(y, x)[2] = inimg_ptr[idx].r;
				outimg.at<cv::Vec4b>(y, x)[3] = inimg_ptr[idx].a;
			}
	}
}

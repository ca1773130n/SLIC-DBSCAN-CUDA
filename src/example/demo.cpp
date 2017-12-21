#include <sdbscan/VCoreEngine.h>
#include <sdbscan/CVAdapter.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#define CAM_WIDTH 1280
#define CAM_HEIGHT 720
#define NUM_SPIXELS 3600

using namespace sdbscan;

void drawDBSCANGraph(SegResult *segRes, std::vector<Color>& colorSetRGB, cv::Mat& frame) {
	for (int i = 0; i < segRes->G->numV; ++i) {
		for (int j = 0; j < segRes->G->nodes[i].numNeighbors; ++j) {
			int nIdx = segRes->G->Ea[segRes->G->va2[i] + j];
			SegNode *n = &segRes->G->nodes[nIdx];
			SegNode *o = &segRes->G->nodes[i];
			cv::Point p1, p2;
			p1.x = segRes->mSpixelMap->GetData(MEMORYDEVICE_CPU)[i].center.x;
			p1.y = segRes->mSpixelMap->GetData(MEMORYDEVICE_CPU)[i].center.y;
			p2.x = segRes->mSpixelMap->GetData(MEMORYDEVICE_CPU)[nIdx].center.x;
			p2.y = segRes->mSpixelMap->GetData(MEMORYDEVICE_CPU)[nIdx].center.y;
			Color color = colorSetRGB[segRes->G->nodes[i].cluster];
			Vector4u lineColor = segRes->mSpixelMap->GetData(MEMORYDEVICE_CPU)[i].color_info.toUChar();
			line(frame, p1, p2, cv::Scalar(lineColor.x, lineColor.y, lineColor.z, 255), 0.1);
		}
	}
}

void drawClusterMap(SegResult *segRes, std::vector<Color>& colorSetRGB, cv::Mat& clusterIndexMap, cv::Mat& clusterMap) {
	for (int i = 0; i < clusterMap.cols; ++i) {
		for (int j = 0; j < clusterMap.rows; ++j) {
			int b = clusterIndexMap.at<cv::Vec3b>(j, i)[0];
			int g = clusterIndexMap.at<cv::Vec3b>(j, i)[1];
			int r = clusterIndexMap.at<cv::Vec3b>(j, i)[2];
			int spID = (r << 16 | g << 8 | b);
			if (segRes->G->nodes[spID].type == NODE_CORE) {
				int labelIndex = segRes->G->nodes[spID].cluster;
				Color segColor = colorSetRGB[labelIndex];
				clusterMap.at<cv::Vec3b>(j, i)[0] = segColor[0];
				clusterMap.at<cv::Vec3b>(j, i)[1] = segColor[1];
				clusterMap.at<cv::Vec3b>(j, i)[2] = segColor[2];
			}
		}
	}
}

int main(int argc, char *argv[]) {
	cv::VideoCapture *vidCap = nullptr;
	if (argc > 1)
		vidCap = new cv::VideoCapture(argv[1]);
	else
		vidCap = new cv::VideoCapture(0);

	if (!vidCap->isOpened()) {
		std::cerr << "unable to open camera with OpenCV!" << std::endl;
	} else {
		int width = CAM_WIDTH;
		int height = CAM_HEIGHT;
		vidCap->set(CV_CAP_PROP_FRAME_WIDTH, width);
		vidCap->set(CV_CAP_PROP_FRAME_HEIGHT, height);

		VCoreEngine engine;
		engine.initSettings(width, height, NUM_SPIXELS);

		cv::Size frameSize(width, height);
		cv::Mat frame, frameClusterMap, frameClusterIndex, frameNumPixels, frameAvgColor, frameCenters;
		frame.create(frameSize, CV_8UC3);
		frameClusterMap.create(frameSize, CV_8UC3);
		frameClusterIndex.create(frameSize, CV_8UC3);
		frameNumPixels.create(frameSize, CV_8UC3);
		frameAvgColor.create(frameSize, CV_8UC3);
		frameCenters.create(frameSize, CV_8UC3);

		UChar4Image *frameImg = new UChar4Image(Vector2i(width, height), true, false);

		while (vidCap->read(frame)) {
			ConvertMatToUC4ImgRGB(frame, frameImg);
			engine.processFrame(frameImg);
			SegResult *res = engine.drawSegmentationResult();

			ConvertUC4ImgToMatRGB(res->mOutputImg, frame);
			ConvertUC4ImgToMatRGB(res->mClusterImg, frameClusterIndex);
			ConvertUC4ImgToMatRGB(res->mNumPixelsImg, frameNumPixels);
			ConvertUC4ImgToMatRGB(res->mAvgColorImg, frameAvgColor);
			ConvertUC4ImgToMatRGB(res->mCenterImg, frameCenters);

			std::vector<Color>& colorSet = engine.getColorSet();
			drawDBSCANGraph(res, colorSet, frameCenters);
			drawClusterMap(res, colorSet, frameClusterIndex, frameClusterMap);

			cv::imshow("Frame", frame);
			cv::imshow("SLIC", frameAvgColor);
			cv::imshow("SLIC-DBSCAN", frameClusterMap);
			cv::imshow("Cluster Num Pixels", frameNumPixels);
			cv::imshow("Cluster Graph", frameCenters);
			cv::imshow("Cluster Index", frameClusterIndex);

			cv::moveWindow("Frame", 0, 0);
			cv::moveWindow("SLIC", width, 0);
			cv::moveWindow("SLIC-DBSCAN", width * 2, 0);
			cv::moveWindow("Cluster Num Pixels", 0, height);
			cv::moveWindow("Cluster Graph", width, height);
			cv::moveWindow("Cluster Index", width * 2, height);
			cv::waitKey(1);
		}	
	}
	return 0;
}

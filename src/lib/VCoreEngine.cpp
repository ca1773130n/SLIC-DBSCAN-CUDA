#include <sdbscan/VCoreEngine.h>
#include <fstream>
#include <set>

namespace sdbscan {
	void VCoreEngine::initSettings(size_t width, size_t height, size_t numSegments, size_t spixelSize) {
		SegResult *resImages = new SegResult;

		mSettings.mImageSize.x = width;
		mSettings.mImageSize.y = height;
		mSettings.mNumSegments = numSegments;
		mSettings.mSpixelSize = spixelSize;
		mSettings.mCohWeight = 0.6f;
		mSettings.mNumIteration = 5;
		mSettings.mColorSpace = RGB;
		mSettings.mSegmentationMethod = GIVEN_NUM;
		mSettings.mForceConnectivity = true;
		mSettings.mSegColors = new int[mSettings.mNumSegments];

		resImages->mOriginalImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mInputImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mOutputImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mClusterImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mAvgColorImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mNumPixelsImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mCenterImg = new UChar4Image(mSettings.mImageSize, true, true);
		resImages->mNumSegments = mSettings.mNumSegments;
		resImages->G = new SegGraph;
		resImages->mSpixelSize = (int)ceil(sqrtf((mSettings.mImageSize.x * mSettings.mImageSize.y) / (float)mSettings.mNumSegments));

		mSettings.mResImages = resImages;

		generateColorSet(mSettings, mColorSetRGB);
		mSegEngine = new VSegEngineGPU(mSettings);
		mClusterEngine = new VClusterEngine;
	}

	void VCoreEngine::processFrame(UChar4Image* inImage) {
		mSegEngine->performSegmentation(inImage);
	}

	const IntImage * VCoreEngine::getSegmentedMask(void) {
		return mSegEngine->getSegMask();
	}

	SegResult *VCoreEngine::drawSegmentationResult(void) {
		SegResult *res = mSettings.mResImages;
		mClusterEngine->makeGraph(0, 40, mSegEngine->getSpixelSize(), mSegEngine->getSpixelMap()->GetData(MEMORYDEVICE_CPU), res->mNumSegments, res->G);
		mClusterEngine->identifyCluster(res->G);
		mSegEngine->drawSegmentationResult(res);

		return res;
	}

	void VCoreEngine::writeSegResToPGM(const char* fileName) {
		const IntImage* idx_img = mSegEngine->getSegMask();
		int width = idx_img->noDims.x;
		int height = idx_img->noDims.y;
		const int* data_ptr = idx_img->GetData(MEMORYDEVICE_CPU);

		std::ofstream f(fileName, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		f << "P5\n" << width << " " << height << "\n65535\n";

		for (int i = 0; i < height * width; ++i) {
			ushort lable = (ushort)data_ptr[i];
			ushort lable_buffer = (lable << 8 | lable >> 8);
			f.write((const char*)&lable_buffer, sizeof(ushort));
		}
		f.close();
	}

	void VCoreEngine::generateColorSet(const VSettings& settings, std::vector<Color>& colorSetRGB) {
		std::set<int> colorSet;

		for (int i = 0; i < settings.mNumSegments; ++i) {
			while (true) {
				Color color(rand() % 200 + 50, rand() % 200 + 55, rand() % 200 + 50);
				int colorInt = color.toInt();

				if (colorSet.find(colorInt) == colorSet.end()) {
					settings.mSegColors[i] = colorInt;
					colorSet.insert(colorInt);
					colorSetRGB.push_back(color);
					break;
				}
			}
		}
	}
}

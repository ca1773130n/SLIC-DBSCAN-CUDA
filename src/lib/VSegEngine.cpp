#include <sdbscan/VSegEngine.h>

namespace sdbscan {
	VSegEngine::VSegEngine(const VSettings& inSettings) {
		mSettings = inSettings;
	}

	VSegEngine::~VSegEngine() {
		if (mSourceImg != NULL) delete mSourceImg;
		if (mConvertedImg != NULL) delete mConvertedImg;
		if (mIndexImg != NULL) delete mIndexImg;
		if (mSpixelMap != NULL) delete mSpixelMap;
	}

	void VSegEngine::performSegmentation(UChar4Image* inputImg) {
		mSourceImg->SetFrom(inputImg, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CUDA);
		convertImgSpace(mSourceImg, mConvertedImg, mSettings.mColorSpace);

		initClusterCenters();
		findCenterAssociation();

		for (int i = 0; i < mSettings.mNumIteration; i++) {
			updateClusterCenter();
			findCenterAssociation();
		}

		if (mSettings.mForceConnectivity)
			enforceConnectivity();
		cudaThreadSynchronize();
	}
}

#pragma once
#include <sdbscan/VDefines.h>
#include <sdbscan/VSpixelInfo.h>

namespace sdbscan {
	struct VSettings {
		Vector2i mImageSize;
		int mNumSegments;
		int mSpixelSize;
		int mNumIteration;
		float mCohWeight;
		bool mForceConnectivity;
		int *mSegColors;
		SegResult *mResImages;

		COLOR_SPACE mColorSpace;
		SEG_METHOD mSegmentationMethod;
	};
}

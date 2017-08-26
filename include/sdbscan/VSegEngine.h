#pragma once
#include <sdbscan/VDefines.h>
#include <sdbscan/VSettings.h>
#include <sdbscan/VSpixelInfo.h>
#include <thrust/device_vector.h>

namespace sdbscan {
	class VSegEngine {
		public:
			VSegEngine(const VSettings& in_settings);
			virtual ~VSegEngine();

			inline SPixelMap *getSpixelMap(void) {
				return mSpixelMap;
			}

			inline int getSpixelSize(void) {
				return mSpixelSize;
			}

			inline const IntImage* getSegMask() const {
				mIndexImg->updateHostFromDevice();
				return mIndexImg;
			};

			void performSegmentation(UChar4Image* inputImg);
			virtual void drawSegmentationResult(const SegResult *result) = 0;

		protected:
			float mMaxColorDist;
			float mMaxPosDist;
			int *mSegColors;
			int mSpixelSize;

			UChar4Image *mSourceImg;
			Float4Image *mConvertedImg;
			IntImage *mIndexImg;
			IntImage *mTempIndexImg;
			SPixelMap* mSpixelMap;

			VSettings mSettings;

			virtual void convertImgSpace(UChar4Image* inputImg, Float4Image* outputImg, COLOR_SPACE colorSpace) = 0;
			virtual void initClusterCenters() = 0;
			virtual void findCenterAssociation() = 0;
			virtual void updateClusterCenter() = 0;
			virtual void enforceConnectivity() = 0;
	};
}


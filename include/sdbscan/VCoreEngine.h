#pragma once
#include "VSegEngineGPU.h"
#include "VClusterEngine.h"
#include "VColor.h"

namespace sdbscan {
	class VCoreEngine {
		public:
			VCoreEngine() : mSegEngine(NULL), mClusterEngine(NULL) {}
			~VCoreEngine() {
				delete mSegEngine;
				delete mClusterEngine;
				delete mSettings.mResImages;
				delete[] mSettings.mSegColors;
			}

			inline bool initialized(void) {
				return (mSegEngine != NULL);
			}

			inline std::vector<Color>& getColorSet(void) {
				return mColorSetRGB;
			}

			void initSettings(size_t width, size_t height, size_t numSegments, size_t spixelSize = 0);
			void processFrame(UChar4Image* inputImg);
			const IntImage * getSegmentedMask();
			SegResult *drawSegmentationResult();
			void writeSegResToPGM(const char* fileName);
			void generateColorSet(const VSettings& settings, std::vector<Color>& colorSetRGB);

		private:
			VSegEngine* mSegEngine;
			VClusterEngine* mClusterEngine;
			std::vector<Color> mColorSetRGB;
			VSettings mSettings;
	};
}


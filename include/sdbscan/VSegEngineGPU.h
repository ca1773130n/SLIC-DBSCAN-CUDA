#pragma once
#include <sdbscan/VSegEngine.h>

namespace sdbscan {
	class VSegEngineGPU : public VSegEngine {
		public:
			VSegEngineGPU(const VSettings& in_settings);
			~VSegEngineGPU();

			void drawSegmentationResult(const SegResult *res);

		protected:
			void convertImgSpace(UChar4Image* inimg, Float4Image* outimg, COLOR_SPACE color_space);
			void initClusterCenters(void);
			void findCenterAssociation(void);
			void updateClusterCenter(void);
			void enforceConnectivity(void);

			int no_grid_per_center;
			SPixelMap* accum_map;
			IntImage* tmp_idx_img;
			int* segColors;
	};
}

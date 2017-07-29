#pragma once
#include "VDefines.h"

namespace sdbscan {
	class SPixelInfo {
	public:
		Vector2f center;
		Vector2f min;
		Vector2f max;
		Vector4f color_info;
		int id;
		int idColor;
		int no_pixels;
		int no_neighbors;
	};
	typedef ORUtils::Image<SPixelInfo> SPixelMap;

	enum NodeType {
		NODE_CORE,
		NODE_BORDER
	};

	class SegNode {
	public:
		SegNode() : type(NODE_CORE), cluster(-1), visited(false), ePtr(0), numNeighbors(0) {}

		NodeType type;
		int cluster;
		bool visited;
		int ePtr;
		int numNeighbors;
	};

	class SegGraph {
	public:
		SegGraph() : nodes(nullptr), e(nullptr), Ea(nullptr) {}
		~SegGraph() {
			delete[] nodes;
			delete[] Ea;
			cudaFree(e);
		}

		SegNode *nodes;
		std::multimap<int, int> clusterMap;
		thrust::host_vector<int> va2;
		int *e;
		int *Ea;
		int numV;
		int numE;
		int numClusters;
	};

	class SegResult {
	public:
		SegResult() {}
		~SegResult() {
			delete mOriginalImg;
			delete mInputImg;
			delete mOutputImg;
			delete mClusterImg;
			delete mAvgColorImg;
			delete mNumPixelsImg;
			delete mSpixelImgs;
			delete mSpixelMap;
			delete G;
		}

		UChar4Image *mOriginalImg;
		UChar4Image *mInputImg;
		UChar4Image *mOutputImg;
		UChar4Image *mClusterImg;
		UChar4Image *mAvgColorImg;
		UChar4Image *mNumPixelsImg;
		UChar4Image *mCenterImg;
		UChar4ImageArray *mSpixelImgs;
		SPixelMap *mSpixelMap;
		SegGraph *G;
		int mNumSegments;
		int mSpixelSize;
	};
}

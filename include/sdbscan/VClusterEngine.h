#pragma once

#include "VSpixelInfo.h"

namespace sdbscan {
	class VClusterEngine {
		public:
			VClusterEngine() {}
			~VClusterEngine() {}

			void makeGraph(int minPts, float Rs, float Rc, SPixelInfo *si, int numVertices, SegGraph *G);
			void classifyObject(SegGraph *G, int i, int minPts);
			void identifyCluster(SegGraph *G);
			void breadthFirstSearch(SegGraph *G, int i, int clusterID);
	};
}

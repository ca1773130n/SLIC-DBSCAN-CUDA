#include <sdbscan/VClusterEngine.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace sdbscan {
	__global__ void breadthFirstSearch_device(SegNode *N, int numV, int *E, unsigned char *Fa, unsigned char *Xa);
	__global__ void makeGraph_device_step1(SPixelInfo* si, SegNode *nodes, int *Va1, int numV, float Rs, float Rc);
	__global__ void makeGraph_device_step2(SPixelInfo* si, SegNode *nodes, int *Va1, int *Va2, int *e, int numV, float Rs, float Rc);

	__device__ float distanceS(SPixelInfo *si1, SPixelInfo *si2) {
		float diffX = fabs(si1->center.x - si2->center.x);
		float diffY = fabs(si1->center.y - si2->center.y);
		float posDiff = sqrtf(diffX * diffX + diffY * diffY);
		return posDiff;
	}

	__device__ float distanceC(SPixelInfo *si1, SPixelInfo *si2) {
		float diffR = fabs(si1->color_info.r - si2->color_info.r);
		float diffG = fabs(si1->color_info.g - si2->color_info.g);
		float diffB = fabs(si1->color_info.b - si2->color_info.b);
		float colorDiff = sqrtf(diffR * diffR + diffG * diffG + diffB * diffB);
		return colorDiff;
	}

	void VClusterEngine::classifyObject(SegGraph *G, int i, int minPts) {
		if (G->nodes[i].numNeighbors >= minPts)
			G->nodes[i].type = NODE_CORE;
		else G->nodes[i].type = NODE_BORDER;
	}

	void VClusterEngine::makeGraph(int minPts, float Rs, float Rc, SPixelInfo *si, int numVertices, SegGraph *G) {
		G->nodes = new SegNode[numVertices];
		G->numV = numVertices;

		const dim3 block(10, 1);
		const dim3 grid(G->numV / block.x, 1);

		thrust::device_vector<int> Va1(numVertices);
		thrust::device_vector<int> Va2(numVertices);
		SPixelInfo *siPtr;
		SegNode *NPtr;

		ORcudaSafeCall(cudaMalloc<struct SegNode>(&NPtr, sizeof(SegNode) * G->numV));
		ORcudaSafeCall(cudaMalloc<struct SPixelInfo>(&siPtr, sizeof(SPixelInfo) * G->numV));
		ORcudaSafeCall(cudaMemcpy(siPtr, si, sizeof(SPixelInfo) * G->numV, cudaMemcpyHostToDevice));
		ORcudaSafeCall(cudaMemcpy(NPtr, G->nodes, sizeof(SegNode) * G->numV, cudaMemcpyHostToDevice));

		int* Va1Ptr = thrust::raw_pointer_cast(Va1.data());
		int* Va2Ptr = thrust::raw_pointer_cast(Va2.data());
		makeGraph_device_step1 << <grid, block >> >(siPtr, NPtr, Va1Ptr, numVertices, Rs, Rc);

		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		ORcudaSafeCall(cudaMemcpy(G->nodes, NPtr, sizeof(SegNode)* G->numV, cudaMemcpyDeviceToHost));

		thrust::exclusive_scan(Va1.begin(), Va1.end(), Va2.begin());

		G->numE = Va1[G->numV - 1] + Va2[G->numV - 1];
		G->Ea = new int[G->numE];
		ORcudaSafeCall(cudaMalloc<int>(&G->e, sizeof(int)* G->numE));
		makeGraph_device_step2 << <grid, block >> >(siPtr, NPtr, Va1Ptr, Va2Ptr, G->e, numVertices, Rs, Rc);

		cudaThreadSynchronize();
		cudaDeviceSynchronize();
		G->va2.clear();
		G->va2.reserve(G->numV);
		thrust::copy(Va2.begin(), Va2.end(), G->va2.begin());

		ORcudaSafeCall(cudaMemcpy(G->Ea, G->e, sizeof(int)* G->numE, cudaMemcpyDeviceToHost));

		for (int i = 0; i < G->numV; ++i) {
			classifyObject(G, i, minPts);
			G->nodes[i].ePtr = Va2[i];
		}

		ORcudaSafeCall(cudaFree(siPtr));
		ORcudaSafeCall(cudaFree(NPtr));
	}

	__global__ void makeGraph_device_step1(SPixelInfo* si, SegNode *nodes, int *Va1, int numV, float Rs, float Rc) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		SPixelInfo *thisSP = &si[tid];
		SegNode *thisNode = &nodes[tid];

		thisNode->type = NODE_BORDER;
		thisNode->numNeighbors = 0;
		thisNode->visited = false;
		thisNode->ePtr = -1;
		thisNode->cluster = -1;

		Va1[tid] = 0;

		for (int i = 0; i < numV; ++i) {
			float distS = distanceS(thisSP, &si[i]);
			float distC = distanceC(thisSP, &si[i]);
			if (tid != i && distS <= Rs && distC <= Rc) {
				thisNode->numNeighbors++;
			}
		}
		Va1[tid] = thisNode->numNeighbors;
	}

	__global__ void makeGraph_device_step2(SPixelInfo* si, SegNode *nodes, int *Va1, int *Va2, int *e, int numV, float Rs, float Rc) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		SPixelInfo *thisSP = &si[tid];

		int ePtr = Va2[tid];
		for (int i = 0; i < numV; ++i) {
			float distS = distanceS(thisSP, &si[i]);
			float distC = distanceC(thisSP, &si[i]);
			if (tid != i && distS <= Rs && distC <= Rc) {
				e[ePtr++] = i;
			}
		}
	}

	void VClusterEngine::identifyCluster(SegGraph *G) {
		int clusterID = 0;

		for (int i = 0; i < G->numV; ++i) {
			if (!G->nodes[i].visited && G->nodes[i].type == NODE_CORE) {
				G->nodes[i].visited = true;
				G->nodes[i].cluster = clusterID;
				G->clusterMap.insert(std::make_pair(clusterID, i));
				breadthFirstSearch(G, i, clusterID);
				clusterID++;
			}
		}

		G->numClusters = clusterID;
	}

	void VClusterEngine::breadthFirstSearch(SegGraph* G, int v, int clusterID) {
		int inputBytes = sizeof(unsigned char)* G->numV;

		unsigned char *XaPtr;
		unsigned char *FaPtr;
		SegNode *NPtr;
		ORcudaSafeCall(cudaMalloc<unsigned char>(&XaPtr, inputBytes));
		ORcudaSafeCall(cudaMalloc<unsigned char>(&FaPtr, inputBytes));
		ORcudaSafeCall(cudaMalloc<struct SegNode>(&NPtr, sizeof(SegNode)* G->numV));
		ORcudaSafeCall(cudaMemcpy(NPtr, G->nodes, sizeof(SegNode)* G->numV, cudaMemcpyHostToDevice));

		unsigned char *Xa = new unsigned char[G->numV];
		unsigned char *Fa = new unsigned char[G->numV];
		memset(Xa, 0, sizeof(unsigned char)* G->numV);
		memset(Fa, 0, sizeof(unsigned char)* G->numV);
		Fa[v] = 1;
		ORcudaSafeCall(cudaMemcpy(FaPtr, Fa, sizeof(unsigned char)* G->numV, cudaMemcpyHostToDevice));
		ORcudaSafeCall(cudaMemcpy(XaPtr, Xa, sizeof(unsigned char)* G->numV, cudaMemcpyHostToDevice));
		int countFa = 1;

		const dim3 block(10, 1);
		const dim3 grid(G->numV / block.x, 1);

		int countLoop = 0;
		while (countFa > 0) {
			breadthFirstSearch_device << <grid, block >> >(NPtr, G->numV, G->e, FaPtr, XaPtr);
			cudaThreadSynchronize();
			cudaDeviceSynchronize();
			ORcudaSafeCall(cudaMemcpy(Fa, FaPtr, inputBytes, cudaMemcpyDeviceToHost));

			countFa = thrust::count(thrust::device, FaPtr, FaPtr + G->numV, 1);
			countLoop++;
		}

		ORcudaSafeCall(cudaMemcpy(Xa, XaPtr, inputBytes, cudaMemcpyDeviceToHost));

		for (int i = 0; i < G->numV; ++i) {
			if (Xa[i]) {
				G->clusterMap.insert(std::make_pair(clusterID, i));
				G->nodes[i].cluster = clusterID;
				G->nodes[i].visited = true;
				if (G->nodes[i].type != NODE_CORE) {
					G->nodes[i].type = NODE_BORDER;
				}
			}
		}

		ORcudaSafeCall(cudaFree(XaPtr));
		ORcudaSafeCall(cudaFree(FaPtr));
		ORcudaSafeCall(cudaFree(NPtr));
		delete[] Xa;
		delete[] Fa;
	}

	__global__ void breadthFirstSearch_device(SegNode *N, int numV, int *EPtr, unsigned char *Fa, unsigned char *Xa) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (Fa[tid]) {
			Fa[tid] = 0;
			Xa[tid] = 1;

			int ePtr = N[tid].ePtr;
			for (int i = 0; i < N[tid].numNeighbors; ++i) {
				int nid = EPtr[ePtr + i];
				Fa[nid] = 1 - Xa[nid];
			}
		}
	}
}


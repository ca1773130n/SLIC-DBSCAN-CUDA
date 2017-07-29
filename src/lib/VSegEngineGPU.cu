// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <sdbscan/VSegEngineGPU.h>
#include <sdbscan/VSegEngineShared.h>
#include <iostream>
#include <thrust/device_vector.h>

namespace sdbscan {
	// ----------------------------------------------------
	//
	//	kernel function defines
	//
	// ----------------------------------------------------

	__global__ void ConvertImgSpace_device(const Vector4u* inimg, Vector4f* outimg, Vector2i mImageSize, COLOR_SPACE mColorSpace);

	__global__ void EnforceConnectivity_device(const int* in_mIndexImg, int* out_mIndexImg, Vector2i mImageSize);

	__global__ void InitClusterCenters_device(const int* mSegColors, const Vector4f* inimg, SPixelInfo* accum_map, SPixelInfo* out_spixel, Vector2i map_size, Vector2i mImageSize, int mSpixelSize);

	__global__ void FindCenterAssociation_device(const Vector4f* inimg, const SPixelInfo* in_mSpixelMap, int* out_mIndexImg, Vector2i map_size, Vector2i mImageSize, int mSpixelSize, float weight, float mMaxPosDist, float mMaxColorDist);

	__global__ void UpdateClusterCenter_device(const Vector4f* inimg, const int* in_mIndexImg, SPixelInfo* accum_map, Vector2i map_size, Vector2i mImageSize, int mSpixelSize, int no_blocks_per_line);

	__global__ void FinalizeReductionResult_device(const SPixelInfo* accum_map, SPixelInfo* spixel_list, Vector2i map_size, int no_blocks_per_spixel);

	__global__ void DrawSegmentationResult_device(const SPixelInfo* in_mSpixelMap, const int mSpixelSize, const int* mIndexImg, Vector4u* orgimg, Vector4u* sourceimg, Vector4u* outimg, Vector4u* clusterimg, Vector4u* avgcolorimg, Vector4u* numpixelsimg, Vector4u* centerimg, Vector4u** ptrs, Vector2i mImageSize);

	// ----------------------------------------------------
	//
	//	host function implementations
	//
	// ----------------------------------------------------

	VSegEngineGPU::VSegEngineGPU(const VSettings& inputSettings) : VSegEngine(inputSettings)
	{
		mSourceImg = new UChar4Image(inputSettings.mImageSize, true, true);
		mConvertedImg = new Float4Image(inputSettings.mImageSize, true, true);
		mIndexImg = new IntImage(inputSettings.mImageSize, true, true);
		mTempIndexImg = new IntImage(inputSettings.mImageSize, true, true);
		ORcudaSafeCall(cudaMalloc<int>(&mSegColors, sizeof(int)* inputSettings.mNumSegments));
		ORcudaSafeCall(cudaMemcpy(mSegColors, inputSettings.mSegColors, inputSettings.mNumSegments * sizeof(int), cudaMemcpyHostToDevice));
		if (inputSettings.mSegmentationMethod == GIVEN_NUM)
		{
			float cluster_size = (float)(inputSettings.mImageSize.x * inputSettings.mImageSize.y) / (float)inputSettings.mNumSegments;
			mSpixelSize = (int)ceil(sqrtf(cluster_size));
			inputSettings.mResImages->mSpixelImgs = new UChar4ImageArray(inputSettings.mNumSegments, Vector2i(mSpixelSize * 3, mSpixelSize * 3), true, true);
		}
		else
		{
			mSpixelSize = inputSettings.mSpixelSize;
		}

		int spixel_per_col = (int)ceil(inputSettings.mImageSize.x / mSpixelSize);
		int spixel_per_row = (int)ceil(inputSettings.mImageSize.y / mSpixelSize);

		Vector2i map_size = Vector2i(spixel_per_col, spixel_per_row);
		mSpixelMap = new SPixelMap(map_size, true, true);
		inputSettings.mResImages->mSpixelMap = mSpixelMap;

		float total_pixel_to_search = (float)(mSpixelSize * mSpixelSize * 9);
		no_grid_per_center = (int)ceil(total_pixel_to_search / (float)(BLOCK_DIM * BLOCK_DIM));

		map_size.x *= no_grid_per_center;
		accum_map = new ORUtils::Image<SPixelInfo>(map_size, true, true);

		// normalizing factors
		mMaxPosDist = 1.0f / (1.4242f * mSpixelSize); // sqrt(2) * mSpixelSize
		switch (inputSettings.mColorSpace)
		{
			case RGB:
				mMaxColorDist = 5.0f / (1.7321f * 255);
				break;
			case XYZ:
				mMaxColorDist = 5.0f / 1.7321f;
				break;
			case CIELAB:
				mMaxColorDist = 15.0f / (1.7321f * 128);
				break;
		}

		mMaxColorDist *= mMaxColorDist;
		mMaxPosDist *= mMaxPosDist;
	}

	VSegEngineGPU::~VSegEngineGPU()
	{
		delete accum_map;
		ORcudaSafeCall(cudaFree(mSegColors));
	}


	void VSegEngineGPU::convertImgSpace(UChar4Image* inimg, Float4Image* outimg, COLOR_SPACE mColorSpace)
	{
		Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CUDA);
		Vector4f* outimg_ptr = outimg->GetData(MEMORYDEVICE_CUDA);
		Vector2i mImageSize = inimg->noDims;

		dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 gridSize((int)ceil((float)mImageSize.x / (float)blockSize.x), (int)ceil((float)mImageSize.y / (float)blockSize.y));

		ConvertImgSpace_device << <gridSize, blockSize >> >(inimg_ptr, outimg_ptr, mImageSize, mColorSpace);

	}

	void VSegEngineGPU::initClusterCenters()
	{
		SPixelInfo* accum_map_ptr = accum_map->GetData(MEMORYDEVICE_CUDA);
		SPixelInfo* spixel_list = mSpixelMap->GetData(MEMORYDEVICE_CUDA);
		Vector4f* img_ptr = mConvertedImg->GetData(MEMORYDEVICE_CUDA);

		Vector2i map_size = mSpixelMap->noDims;
		Vector2i mImageSize = mConvertedImg->noDims;

		dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 gridSize((int)ceil((float)map_size.x / (float)blockSize.x), (int)ceil((float)map_size.y / (float)blockSize.y));

		InitClusterCenters_device << <gridSize, blockSize >> >(mSegColors, img_ptr, accum_map_ptr, spixel_list, map_size, mImageSize, mSpixelSize);
	}

	void VSegEngineGPU::findCenterAssociation()
	{
		SPixelInfo* spixel_list = mSpixelMap->GetData(MEMORYDEVICE_CUDA);
		Vector4f* img_ptr = mConvertedImg->GetData(MEMORYDEVICE_CUDA);
		int* idx_ptr = mIndexImg->GetData(MEMORYDEVICE_CUDA);

		Vector2i map_size = mSpixelMap->noDims;
		Vector2i mImageSize = mConvertedImg->noDims;

		dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 gridSize((int)ceil((float)mImageSize.x / (float)blockSize.x), (int)ceil((float)mImageSize.y / (float)blockSize.y));

		FindCenterAssociation_device << <gridSize, blockSize >> >(img_ptr, spixel_list, idx_ptr, map_size, mImageSize, mSpixelSize, mSettings.mCohWeight, mMaxPosDist, mMaxColorDist);
	}

	void VSegEngineGPU::updateClusterCenter()
	{
		SPixelInfo* accum_map_ptr = accum_map->GetData(MEMORYDEVICE_CUDA);
		SPixelInfo* spixel_list_ptr = mSpixelMap->GetData(MEMORYDEVICE_CUDA);
		Vector4f* img_ptr = mConvertedImg->GetData(MEMORYDEVICE_CUDA);
		int* idx_ptr = mIndexImg->GetData(MEMORYDEVICE_CUDA);

		Vector2i map_size = mSpixelMap->noDims;
		Vector2i mImageSize = mConvertedImg->noDims;

		int no_blocks_per_line = mSpixelSize * 3 / BLOCK_DIM;

		dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 gridSize(map_size.x, map_size.y, no_grid_per_center);

		UpdateClusterCenter_device << <gridSize, blockSize >> >(img_ptr, idx_ptr, accum_map_ptr, map_size, mImageSize, mSpixelSize, no_blocks_per_line);

		dim3 gridSize2(map_size.x, map_size.y);

		FinalizeReductionResult_device << <gridSize2, blockSize >> >(accum_map_ptr, spixel_list_ptr, map_size, no_grid_per_center);
		mSpixelMap->updateHostFromDevice();
	}

	void VSegEngineGPU::enforceConnectivity()
	{
		int* idx_ptr = mIndexImg->GetData(MEMORYDEVICE_CUDA);
		int* tmp_idx_ptr = mTempIndexImg->GetData(MEMORYDEVICE_CUDA);
		Vector2i mImageSize = mIndexImg->noDims;

		dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 gridSize((int)ceil((float)mImageSize.x / (float)blockSize.x), (int)ceil((float)mImageSize.y / (float)blockSize.y));

		EnforceConnectivity_device << <gridSize, blockSize >> >(idx_ptr, tmp_idx_ptr, mImageSize);
		EnforceConnectivity_device << <gridSize, blockSize >> >(tmp_idx_ptr, idx_ptr, mImageSize);
	}

	void VSegEngineGPU::drawSegmentationResult(const SegResult *res)
	{
		SPixelInfo* sPixelListPtr = mSpixelMap->GetData(MEMORYDEVICE_CUDA);
		Vector4u* orgImgPtr = res->mOriginalImg->GetData(MEMORYDEVICE_CUDA);
		Vector4u* inImgPtr = mSourceImg->GetData(MEMORYDEVICE_CUDA);
		Vector4u* outImgPtr = res->mOutputImg->GetData(MEMORYDEVICE_CUDA);
		Vector4u* clusterImgPtr = res->mClusterImg->GetData(MEMORYDEVICE_CUDA);
		Vector4u* avgColorImgPtr = res->mAvgColorImg->GetData(MEMORYDEVICE_CUDA);
		Vector4u* numPixelsImgPtr = res->mNumPixelsImg->GetData(MEMORYDEVICE_CUDA);
		Vector4u* centerImgPtr = res->mCenterImg->GetData(MEMORYDEVICE_CUDA);
		int* idxImgPtr = mIndexImg->GetData(MEMORYDEVICE_CUDA);

		Vector2i mImageSize = mIndexImg->noDims;

		dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 gridSize((int)ceil((float)mImageSize.x / (float)blockSize.x), (int)ceil((float)mImageSize.y / (float)blockSize.y));
		Vector4u** spixels = thrust::raw_pointer_cast(&res->mSpixelImgs->getPtrs()[0]);

		for (size_t i = 0; i < res->mNumSegments; ++i)
			res->mSpixelImgs->getImage(i)->Clear();

		DrawSegmentationResult_device << <gridSize, blockSize >> >(sPixelListPtr, res->mSpixelSize, idxImgPtr, orgImgPtr, inImgPtr, outImgPtr, clusterImgPtr, avgColorImgPtr, numPixelsImgPtr, centerImgPtr, spixels, mImageSize);

		cudaThreadSynchronize();

		res->mOutputImg->updateHostFromDevice();
		res->mClusterImg->updateHostFromDevice();
		res->mAvgColorImg->updateHostFromDevice();
		res->mNumPixelsImg->updateHostFromDevice();
		res->mCenterImg->updateHostFromDevice();
		for (size_t i = 0; i < res->mNumSegments; ++i)
			res->mSpixelImgs->getImage(i)->updateHostFromDevice();
	}

	// ----------------------------------------------------
	//
	//	device function implementations
	//
	// ----------------------------------------------------

	__global__ void ConvertImgSpace_device(const Vector4u* inimg, Vector4f* outimg, Vector2i mImageSize, COLOR_SPACE mColorSpace)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x > mImageSize.x - 1 || y > mImageSize.y - 1) return;

		ConvertImgSpace_shared(inimg, outimg, mImageSize, x, y, mColorSpace);

	}

	__global__ void DrawSegmentationResult_device(const SPixelInfo* in_mSpixelMap, const int mSpixelSize, const int* mIndexImg, Vector4u* orgimg, Vector4u* sourceimg, Vector4u* outimg, Vector4u* clusterimg, Vector4u* avgcolorimg, Vector4u* numpixelsimg, Vector4u* centerimg, Vector4u** spimgs, Vector2i mImageSize)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x == 0 || y == 0 || x > mImageSize.x - 2 || y > mImageSize.y - 2) return;

		DrawSuperPixelBoundry_shared(in_mSpixelMap, mSpixelSize, mIndexImg, orgimg, sourceimg, outimg, clusterimg, avgcolorimg, numpixelsimg, centerimg, spimgs, mImageSize, x, y);
	}

	__global__ void InitClusterCenters_device(const int* mSegColors, const Vector4f* inimg, SPixelInfo* accum_map, SPixelInfo* out_spixel, Vector2i map_size, Vector2i mImageSize, int mSpixelSize)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
		int g = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x);
		if (x > map_size.x - 1 || y > map_size.y - 1) return;

		InitClusterCenters_shared(mSegColors, inimg, accum_map, out_spixel, map_size, mImageSize, mSpixelSize, x, y, g);
	}

	__global__ void FindCenterAssociation_device(const Vector4f* inimg, const SPixelInfo* in_mSpixelMap, int* out_mIndexImg, Vector2i map_size, Vector2i mImageSize, int mSpixelSize, float weight, float mMaxPosDist, float mMaxColorDist)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x > mImageSize.x - 1 || y > mImageSize.y - 1) return;

		FindCenterAssociation_shared(inimg, in_mSpixelMap, out_mIndexImg, map_size, mImageSize, mSpixelSize, weight, x, y, blockDim.x, blockDim.y, mMaxPosDist, mMaxColorDist);
	}

	__global__ void UpdateClusterCenter_device(const Vector4f* inimg, const int* in_mIndexImg, SPixelInfo* accum_map, Vector2i map_size, Vector2i mImageSize, int mSpixelSize, int no_blocks_per_line)
	{
		int local_id = threadIdx.y * blockDim.x + threadIdx.x;

		__shared__ Vector4f color_shared[BLOCK_DIM*BLOCK_DIM];
		__shared__ Vector2f xy_shared[BLOCK_DIM*BLOCK_DIM];
		__shared__ int count_shared[BLOCK_DIM*BLOCK_DIM];
		__shared__ bool should_add;

		color_shared[local_id] = Vector4f(0, 0, 0, 0);
		xy_shared[local_id] = Vector2f(0, 0);
		count_shared[local_id] = 0;
		should_add = false;
		__syncthreads();

		int no_blocks_per_spixel = gridDim.z;

		int spixel_id = blockIdx.y * map_size.x + blockIdx.x;

		// compute the relative position in the search window
		int block_x = blockIdx.z % no_blocks_per_line;
		int block_y = blockIdx.z / no_blocks_per_line;

		int x_offset = block_x * BLOCK_DIM + threadIdx.x;
		int y_offset = block_y * BLOCK_DIM + threadIdx.y;

		if (x_offset < mSpixelSize * 3 && y_offset < mSpixelSize * 3)
		{
			// compute the start of the search window
			int x_start = blockIdx.x * mSpixelSize - mSpixelSize;
			int y_start = blockIdx.y * mSpixelSize - mSpixelSize;

			int x_img = x_start + x_offset;
			int y_img = y_start + y_offset;

			if (x_img >= 0 && x_img < mImageSize.x && y_img >= 0 && y_img < mImageSize.y)
			{
				int img_idx = y_img * mImageSize.x + x_img;
				if (in_mIndexImg[img_idx] == spixel_id)
				{
					int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx.z;
					color_shared[local_id] = inimg[img_idx];
					xy_shared[local_id] = Vector2f(x_img, y_img);
					count_shared[local_id] = 1;
					should_add = true;
				}
			}
		}
		__syncthreads();

		if (should_add)
		{
			if (local_id < 128)
			{
				color_shared[local_id] += color_shared[local_id + 128];
				xy_shared[local_id] += xy_shared[local_id + 128];
				count_shared[local_id] += count_shared[local_id + 128];
			}
			__syncthreads();

			if (local_id < 64)
			{
				color_shared[local_id] += color_shared[local_id + 64];
				xy_shared[local_id] += xy_shared[local_id + 64];
				count_shared[local_id] += count_shared[local_id + 64];
			}
			__syncthreads();

			if (local_id < 32)
			{
				color_shared[local_id] += color_shared[local_id + 32];
				color_shared[local_id] += color_shared[local_id + 16];
				color_shared[local_id] += color_shared[local_id + 8];
				color_shared[local_id] += color_shared[local_id + 4];
				color_shared[local_id] += color_shared[local_id + 2];
				color_shared[local_id] += color_shared[local_id + 1];

				xy_shared[local_id] += xy_shared[local_id + 32];
				xy_shared[local_id] += xy_shared[local_id + 16];
				xy_shared[local_id] += xy_shared[local_id + 8];
				xy_shared[local_id] += xy_shared[local_id + 4];
				xy_shared[local_id] += xy_shared[local_id + 2];
				xy_shared[local_id] += xy_shared[local_id + 1];

				count_shared[local_id] += count_shared[local_id + 32];
				count_shared[local_id] += count_shared[local_id + 16];
				count_shared[local_id] += count_shared[local_id + 8];
				count_shared[local_id] += count_shared[local_id + 4];
				count_shared[local_id] += count_shared[local_id + 2];
				count_shared[local_id] += count_shared[local_id + 1];
			}
		}
		__syncthreads();

		if (local_id == 0)
		{
			int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx.z;
			accum_map[accum_map_idx].center = xy_shared[0];
			accum_map[accum_map_idx].color_info = color_shared[0];
			accum_map[accum_map_idx].no_pixels = count_shared[0];
		}
	}

	__global__ void FinalizeReductionResult_device(const SPixelInfo* accum_map, SPixelInfo* spixel_list, Vector2i map_size, int no_blocks_per_spixel)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x > map_size.x - 1 || y > map_size.y - 1) return;

		FinalizeReductionResult_shared(accum_map, spixel_list, map_size, no_blocks_per_spixel, x, y);
	}

	__global__ void EnforceConnectivity_device(const int* in_mIndexImg, int* out_mIndexImg, Vector2i mImageSize)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x > mImageSize.x - 1 || y > mImageSize.y - 1) return;

		SupressLocalLable(in_mIndexImg, out_mIndexImg, mImageSize, x, y);
	}
}


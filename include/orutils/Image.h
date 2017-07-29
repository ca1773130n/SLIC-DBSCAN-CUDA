// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <vector>
#include <cassert>
#include "MemoryBlock.h"
#include <thrust/device_vector.h>

#ifndef __METALC__

namespace ORUtils
{
	/** \brief
	Represents images, templated on the pixel type
	*/
	template <typename T>
	class Image : public MemoryBlock < T >
	{
	public:
		/** Size of the image in pixels. */
		Vector2<int> noDims;

		/** Initialize an empty image of the given size, either
		on CPU only or on both CPU and GPU.
		*/
		Image(Vector2<int> noDims, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
			: MemoryBlock<T>(noDims.x * noDims.y, allocate_CPU, allocate_CUDA, metalCompatible)
		{
			this->noDims = noDims;
		}

		Image(bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
			: MemoryBlock<T>(1, allocate_CPU, allocate_CUDA, metalCompatible)
		{
			this->noDims = Vector2<int>(1, 1);  //TODO - make nicer
		}

		Image(Vector2<int> noDims, MemoryDeviceType memoryType)
			: MemoryBlock<T>(noDims.x * noDims.y, memoryType)
		{
			this->noDims = noDims;
		}

		/** Resize an image, loosing all old image data.
		Essentially any previously allocated data is
		released, new memory is allocated.
		*/
		void ChangeDims(Vector2<int> newDims)
		{
			if (newDims != noDims)
			{
				this->noDims = newDims;

				bool allocate_CPU = this->isAllocated_CPU;
				bool allocate_CUDA = this->isAllocated_CUDA;
				bool metalCompatible = this->isMetalCompatible;

				this->Free();
				this->Allocate(newDims.x * newDims.y, allocate_CPU, allocate_CUDA, metalCompatible);
			}
		}

		// Suppress the default copy constructor and assignment operator
		Image(const Image&);
		Image& operator=(const Image&);
	};

	template <typename T>
	class ImageArray {
	public:
		ImageArray(int numImages, Vector2<int> noDims, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true) {
			for (int i = 0; i < numImages; ++i)
				mImages.push_back(new Image<T>(noDims, allocate_CPU, allocate_CUDA, metalCompatible));
			//mPtrs = new DEVICEPTR(T)*[numImages];
		}

		~ImageArray() {
			size_t numImages = mImages.size();
			for (size_t i = 0; i < numImages; ++i)
				delete mImages[i];
			mImages.clear();
		}

		inline size_t size(void) {
			return mImages.size();
		}

		Image<T> *getImage(size_t index) {
			assert(index >= 0 && index < mImages.size());
			return mImages[index];
		}

		inline thrust::device_vector<DEVICEPTR(T) *> &getPtrs(void) {
			size_t numImages = mImages.size();
			for (size_t i = 0; i < numImages; ++i)
				mPtrs.push_back(mImages[i]->GetData(MEMORYDEVICE_CUDA));
			return mPtrs;
		}

	protected:
		std::vector<Image<T> *> mImages;
		thrust::device_vector<DEVICEPTR(T) *> mPtrs;
	};
}

#endif

#pragma once

#include "../Signal.h"
#include "VColor.h"

namespace mc {
	namespace sensory {
		namespace vision {
			class ImageSignalRGB8 : public Signal
			{
			public:
				ImageSignalRGB8(SensorType sensorType, unsigned char *rawBytes, SignalDataType dataType, Dim& dims)
					: mColorFormat(COLOR_RGB_888), Signal(sensorType, dataType, dims, NumChannels(COLOR_RGB_888)) {
					mImage = new UChar3Image(Vector2i(dims[0], dims[1]), true, true);
					Vector3u *ptr = mImage->GetData(MEMORYDEVICE_CPU);
					
					for (size_t h = 0; h < mImage->noDims.y; ++h) {
						for (size_t w = 0; w < mImage->noDims.x; ++w) {
							int targetIdx = w + h * mImage->noDims.x;
							int sourceIdx = 3 * (w + h * mImage->noDims.x);
							ptr[sourceIdx].r = rawBytes[targetIdx];
							ptr[sourceIdx].g = rawBytes[targetIdx + 1];
							ptr[sourceIdx].b = rawBytes[targetIdx + 2];
						}
					}

					mImage->UpdateDeviceFromHost();
				}

				UChar3Image *getImage(void) {
					return mImage;
				}
				
				virtual unsigned char *getRawBytes(void) const {
					return reinterpret_cast<unsigned char *>(mImage->GetData(MEMORYDEVICE_CPU));
				}

			protected:
				enum ColorFormat mColorFormat;
				UChar3Image *mImage;
			};

			class ImageSignalRGBA8 : public Signal
			{
			public:
				ImageSignalRGBA8(SensorType sensorType, SignalDataType dataType, Dim& dims)
					: mColorFormat(COLOR_RGBA_8888), Signal(sensorType, dataType, dims, NumChannels(COLOR_RGBA_8888)) {
				}

				ImageSignalRGBA8(SensorType sensorType, SignalDataType dataType, Dim& dims, UChar4Image *image)
					: ImageSignalRGBA8(sensorType, dataType, dims) {
					mImage = image;
				}

				ImageSignalRGBA8(SensorType sensorType, unsigned char *rawBytes, SignalDataType dataType, Dim& dims)
					: ImageSignalRGBA8(sensorType, dataType, dims) {
					mImage = new UChar4Image(Vector2i(dims[0], dims[1]), true, true);
					Vector4u *ptr = mImage->GetData(MEMORYDEVICE_CPU);

					for (size_t y = 0; y < mImage->noDims.y; ++y) {
						for (size_t x = 0; x < mImage->noDims.x; ++x) {
							int targetIdx = 4 * (x + y * mImage->noDims.x);
							int sourceIdx = x + y * mImage->noDims.x;
							ptr[sourceIdx].b = rawBytes[targetIdx];
							ptr[sourceIdx].g = rawBytes[targetIdx + 1];
							ptr[sourceIdx].r = rawBytes[targetIdx + 2];
							ptr[sourceIdx].a = rawBytes[targetIdx + 3];
						}
					}

					mImage->UpdateDeviceFromHost();
				}

				UChar4Image *getImage(void) {
					return mImage;
				}
			
				virtual unsigned char *getRawBytes(void) const {
					return reinterpret_cast<unsigned char *>(mImage->GetData(MEMORYDEVICE_CPU));
				}

			protected:
				enum ColorFormat mColorFormat;
				UChar4Image *mImage;
			};
		}
	}
}
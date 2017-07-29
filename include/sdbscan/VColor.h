#pragma once
#include <vector>

namespace sdbscan {
	enum ColorFormat {
		COLOR_RGB_888,
		COLOR_RGBA_8888,
		COLOR_RGB_FLOAT,
		COLOR_RGBA_FLOAT,
		COLOR_YUV_420,
		COLOR_YUV_422,
		COLOR_INVALID
	};

	inline int NumChannels(enum ColorFormat format) {
		switch (format) {
			case COLOR_RGB_888:
			case COLOR_RGB_FLOAT:
				return 3;
			case COLOR_RGBA_8888:
			case COLOR_RGBA_FLOAT:
				return 4;
		}
		return 0;
	}

	inline int NumBytesPerChannel(enum ColorFormat format) {
		switch (format) {
			case COLOR_RGB_888:
			case COLOR_RGBA_8888:
				return sizeof(unsigned char);
			case COLOR_RGB_FLOAT:
			case COLOR_RGBA_FLOAT:
				return sizeof(float);
		}
		return 0;
	}

	class Color {
		public:
			Color() {}
			Color(unsigned char r, unsigned char g, unsigned char b) {
				mChannels.push_back(r);
				mChannels.push_back(g);
				mChannels.push_back(b);
			}

			ColorFormat getFormat(void) {
				return mFormat;
			}

			float intensity(void) {
				float sum = 0;
				for (unsigned char c : mChannels) {
					sum += c * c;
				}
				return sqrtf(sum);
			}

			void addChannel(unsigned char c) {
				mChannels.push_back(c);
			}

			int toInt(void) {
				return (mChannels[0] << 16) | (mChannels[1] << 8) | mChannels[2];
			}

			unsigned char operator[](int index) {
				return mChannels[index];
			}

		protected:
			ColorFormat mFormat;
			std::vector<unsigned char> mChannels;
	};
}

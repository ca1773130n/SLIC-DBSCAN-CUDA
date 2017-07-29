#pragma once

#include <orutils/Vector.h>
#include <orutils/Matrix.h>
#include <orutils/Image.h>
#include <orutils/MathUtils.h>
#include <orutils/MemoryBlock.h>
#include <thrust/host_vector.h>
#include <map>

//------------------------------------------------------
// 
// Compile time GPU Settings, don't touch it!
//
//------------------------------------------------------

#ifndef BLOCK_DIM
#define BLOCK_DIM		16
#endif

//------------------------------------------------------
// 
// math defines
//
//------------------------------------------------------

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef class ORUtils::Matrix3<float> Matrix3f;
typedef class ORUtils::Matrix4<float> Matrix4f;

typedef class ORUtils::Vector2<short> Vector2s;
typedef class ORUtils::Vector2<int> Vector2i;
typedef class ORUtils::Vector2<float> Vector2f;
typedef class ORUtils::Vector2<double> Vector2d;

typedef class ORUtils::Vector3<short> Vector3s;
typedef class ORUtils::Vector3<double> Vector3d;
typedef class ORUtils::Vector3<int> Vector3i;
typedef class ORUtils::Vector3<uint> Vector3ui;
typedef class ORUtils::Vector3<uchar> Vector3u;
typedef class ORUtils::Vector3<float> Vector3f;

typedef class ORUtils::Vector4<float> Vector4f;
typedef class ORUtils::Vector4<int> Vector4i;
typedef class ORUtils::Vector4<short> Vector4s;
typedef class ORUtils::Vector4<uchar> Vector4u;

//------------------------------------------------------
// 
// image defines
//
//------------------------------------------------------

typedef  ORUtils::Image<Vector4f> Float4Image;
typedef  ORUtils::Image<int> IntImage;
typedef  ORUtils::Image<Vector4u> UChar4Image;
typedef  ORUtils::Image<Vector3u> UChar3Image;
typedef  ORUtils::ImageArray<Vector4u> UChar4ImageArray;

//------------------------------------------------------
// 
// Other defines
//
//------------------------------------------------------

typedef enum
{
	CIELAB = 0,
	XYZ,
	RGB
} COLOR_SPACE;

typedef enum
{
	GIVEN_NUM = 0,
	GIVEN_SIZE

} SEG_METHOD;

#ifndef DEBUGBREAK
#define DEBUGBREAK \
{ \
	int ryifrklaeybfcklarybckyar = 0; \
	ryifrklaeybfcklarybckyar++; \
}
#endif

#pragma once
#include <sdbscan/VDefines.h>
#include <sdbscan/VSpixelInfo.h>
#include <map>

namespace sdbscan {
	_CPU_AND_GPU_CODE_ inline void RGBtoXYZ(const Vector4u& pix_in, Vector4f& pix_out)
	{
		float _b = (float)pix_in.x * 0.0039216f;
		float _g = (float)pix_in.y * 0.0039216f;
		float _r = (float)pix_in.z * 0.0039216f;

		pix_out.x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
		pix_out.y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
		pix_out.z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

	}

	_CPU_AND_GPU_CODE_ inline void RGBtoCIELab(const Vector4u& pix_in, Vector4f& pix_out)
	{
		float _b = (float)pix_in.x * 0.0039216f;
		float _g = (float)pix_in.y * 0.0039216f;
		float _r = (float)pix_in.z * 0.0039216f;

		float x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
		float y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
		float z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

		float epsilon = 0.008856f;	//actual CIE standard
		float kappa = 903.3f;		//actual CIE standard

		float Xr = 0.950456f;	//reference white
		float Yr = 1.0f;		//reference white
		float Zr = 1.088754f;	//reference white

		float xr = x / Xr;
		float yr = y / Yr;
		float zr = z / Zr;

		float fx, fy, fz;
		if (xr > epsilon)	fx = pow(xr, 1.0f / 3.0f);
		else				fx = (kappa*xr + 16.0f) / 116.0f;
		if (yr > epsilon)	fy = pow(yr, 1.0f / 3.0f);
		else				fy = (kappa*yr + 16.0f) / 116.0f;
		if (zr > epsilon)	fz = pow(zr, 1.0f / 3.0f);
		else				fz = (kappa*zr + 16.0f) / 116.0f;

		pix_out.x = 116.0f*fy - 16.0f;
		pix_out.y = 500.0f*(fx - fy);
		pix_out.z = 200.0f*(fy - fz);
	}

	_CPU_AND_GPU_CODE_ inline void ConvertImgSpace_shared(const Vector4u* inimg, Vector4f* outimg, const Vector2i& img_size, int x, int y, const COLOR_SPACE& color_space)
	{
		int idx = y * img_size.x + x;

		switch (color_space)
		{
			case RGB:
				outimg[idx].x = inimg[idx].x;
				outimg[idx].y = inimg[idx].y;
				outimg[idx].z = inimg[idx].z;
				break;
			case XYZ:
				RGBtoXYZ(inimg[idx], outimg[idx]);
				break;
			case CIELAB:
				RGBtoCIELab(inimg[idx], outimg[idx]);
				break;
		}
	}

	_CPU_AND_GPU_CODE_ inline void InitClusterCenters_shared(const int* segColors, const Vector4f* inimg, SPixelInfo* accum_map, SPixelInfo* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size, int x, int y, int g)
	{
		int cluster_idx = y * map_size.x + x;

		int img_x = x * spixel_size + spixel_size / 2;
		int img_y = y * spixel_size + spixel_size / 2;

		img_x = img_x >= img_size.x ? (x * spixel_size + img_size.x) / 2 : img_x;
		img_y = img_y >= img_size.y ? (y * spixel_size + img_size.y) / 2 : img_y;

		// TODO: go one step towards gradients direction

		out_spixel[cluster_idx].id = cluster_idx;
		out_spixel[cluster_idx].idColor = segColors[cluster_idx];
		out_spixel[cluster_idx].center = Vector2f((float)img_x, (float)img_y);
		out_spixel[cluster_idx].color_info = inimg[img_y*img_size.x + img_x];

		out_spixel[cluster_idx].no_pixels = 0;
	}

	_CPU_AND_GPU_CODE_ inline float ComputeSlicDistance(const Vector4f& pix, int x, int y, const SPixelInfo& center_info, float weight, float normalizer_xy, float normalizer_color)
	{
		float dcolor = (pix.x - center_info.color_info.x)*(pix.x - center_info.color_info.x)
			+ (pix.y - center_info.color_info.y)*(pix.y - center_info.color_info.y)
			+ (pix.z - center_info.color_info.z)*(pix.z - center_info.color_info.z);

		float dxy = (x - center_info.center.x) * (x - center_info.center.x)
			+ (y - center_info.center.y) * (y - center_info.center.y);


		float retval = dcolor * normalizer_color + weight * dxy * normalizer_xy;
		return sqrtf(retval);
	}

	_CPU_AND_GPU_CODE_ inline void FindCenterAssociation_shared(const Vector4f* inimg, const SPixelInfo* in_spixel_map, int* out_idx_img, Vector2i map_size, Vector2i img_size, int spixel_size, float weight, int x, int y, int blockDimX, int blockDimY, float max_xy_dist, float max_color_dist)
	{
		int idx_img = y * img_size.x + x;

		int ctr_x = x / spixel_size;
		int ctr_y = y / spixel_size;

		int minidx = -1;
		float dist = 999999.9999f;

		// search 3x3 neighborhood
		for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
		{
			int ctr_x_check = ctr_x + j;
			int ctr_y_check = ctr_y + i;
			if (ctr_x_check >= 0 && ctr_y_check >= 0 && ctr_x_check < map_size.x && ctr_y_check < map_size.y)
			{
				int ctr_idx = ctr_y_check*map_size.x + ctr_x_check;
				float cdist = ComputeSlicDistance(inimg[idx_img], x, y, in_spixel_map[ctr_idx], weight, max_xy_dist, max_color_dist);
				if (cdist < dist)
				{
					dist = cdist;
					minidx = in_spixel_map[ctr_idx].id;
				}
			}
		}

		if (minidx >= 0) out_idx_img[idx_img] = minidx;
	}

	_CPU_AND_GPU_CODE_ inline Vector3u GetRGBFromI(unsigned int RGBint)
	{
		unsigned int blue = RGBint & 255;
		unsigned int green = RGBint / 256;
		unsigned int red = RGBint / 65536;
		return Vector3u(red, green, blue);
	}

	_CPU_AND_GPU_CODE_ Vector3u IntToRGB(int index) {
		return Vector3u((index >> 16) & 0xFF, (index >> 8) & 0xFF, index & 0xFF);
	}

	_CPU_AND_GPU_CODE_ int RgbToInt(Vector3u rgb) {
		return (rgb.r << 16 | rgb.g << 8 | rgb.b);
	}

	_CPU_AND_GPU_CODE_ inline void DrawSuperPixelBoundry_shared(const SPixelInfo* spixel_list, const int spixel_size, const int* idx_img, Vector4u* orgimg, Vector4u* sourceimg, Vector4u* outimg, Vector4u* clusterimg, Vector4u* avgcolorimg, Vector4u* numpixelsimg, Vector4u* centerimg, Vector4u**  spimgs, Vector2i img_size, int x, int y)
	{
		int idx = y * img_size.x + x;
		int orgx = spixel_list[idx_img[idx]].center.x - spixel_size / 2 * 3;
		int orgy = spixel_list[idx_img[idx]].center.y - spixel_size / 2 * 3;

		outimg[idx] = sourceimg[idx];

		int diffx = x - orgx;
		int diffy = y - orgy;
		int spidx = spixel_size * 3 * diffy + diffx;
		spimgs[idx_img[idx]][spidx] = sourceimg[idx];
		spimgs[idx_img[idx]][spidx].a = 255;

		clusterimg[idx] = Vector4u(GetRGBFromI(idx_img[idx]), 255);

		if (idx_img[idx] != idx_img[idx + 1]
				|| idx_img[idx] != idx_img[idx - 1]
				|| idx_img[idx] != idx_img[(y - 1)*img_size.x + x]
				|| idx_img[idx] != idx_img[(y + 1)*img_size.x + x])
		{
			avgcolorimg[idx] = Vector4u(0, 0, 0, 0);
		}
		else
		{
			avgcolorimg[idx] = spixel_list[idx_img[idx]].color_info.toUChar();
		}

		numpixelsimg[idx] = spixel_list[idx_img[idx]].no_pixels;
		if (x == (int)spixel_list[idx_img[idx]].center.x && y == (int)spixel_list[idx_img[idx]].center.y) {
			centerimg[idx] = Vector4u(255, 255, 255, 255);
		}
		else {
			centerimg[idx] = Vector4u(0, 0, 0, 0);
		}
	}

	_CPU_AND_GPU_CODE_ inline void FinalizeReductionResult_shared(const SPixelInfo* accum_map, SPixelInfo* spixel_list, Vector2i map_size, int no_blocks_per_spixel, int x, int y)
	{
		int spixel_idx = y * map_size.x + x;

		spixel_list[spixel_idx].center = Vector2f(0, 0);
		spixel_list[spixel_idx].color_info = Vector4f(0, 0, 0, 0);
		spixel_list[spixel_idx].no_pixels = 0;

		for (int i = 0; i < no_blocks_per_spixel; i++)
		{
			int accum_list_idx = spixel_idx * no_blocks_per_spixel + i;

			spixel_list[spixel_idx].center += accum_map[accum_list_idx].center;
			spixel_list[spixel_idx].color_info += accum_map[accum_list_idx].color_info;
			spixel_list[spixel_idx].no_pixels += accum_map[accum_list_idx].no_pixels;
			spixel_list[spixel_idx].no_neighbors = no_blocks_per_spixel;
		}

		if (spixel_list[spixel_idx].no_pixels != 0)
		{
			spixel_list[spixel_idx].center /= (float)spixel_list[spixel_idx].no_pixels;
			spixel_list[spixel_idx].color_info /= (float)spixel_list[spixel_idx].no_pixels;
		}
	}

	_CPU_AND_GPU_CODE_ inline void SupressLocalLable(const int* in_idx_img, int* out_idx_img, Vector2i img_size, int x, int y)
	{
		int clable = in_idx_img[y*img_size.x + x];

		// don't suppress boundary
		if (x <= 1 || y <= 1 || x >= img_size.x - 2 || y >= img_size.y - 2)
		{
			out_idx_img[y*img_size.x + x] = clable;
			return;
		}

		int diff_count = 0;
		int diff_lable = -1;

		for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++)
		{
			int nlable = in_idx_img[(y + j)*img_size.x + (x + i)];
			if (nlable != clable)
			{
				diff_lable = nlable;
				diff_count++;
			}
		}

		if (diff_count >= 16)
			out_idx_img[y*img_size.x + x] = diff_lable;
		else
			out_idx_img[y*img_size.x + x] = clable;
	}
}

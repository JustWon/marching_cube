/**
/file		cu_kinectfusion_v1.h
\brief		Cuda KIENCT FUSION library

\author		Seong-Oh Lee
\version	1.0
\date		2012.06.25
*/

#ifndef _CU_KIENCTFUSION_V1_H
#define _CU_KIENCTFUSION_V1_H

#include "structure.h" //inyeop
#include <helper_math.h>
#include "cu_device.h"
#include "cu_geometry_functions.h"

#define CG_KLT_TRACKER
#ifdef CG_KLT_TRACKER
#define GLEW_STATIC
//#include "cg_klt_tracker.hpp"
#endif

//#include "new_well/geometry/registration3d.h"
//#include "cu_vector_functions.h"
//#include "cu_geometry_functions.h"
//#include "cu_matrix.hpp"
//#include "new_well/geometry/camera.h"

#define KINECT_VIDEO_LEVEL	3
#define KINECT_DEPTH_LEVEL	3
#define KINECT_DEPTH_MIN	0.0f ///< meter
#define KINECT_DEPTH_MAX	2.0f ///< meter

// Device
#define DEVICE_OPENNI
//#define DEVICE_KINECT
//#define DEVICE_FREENECT

static const char VOLUME_FORMAT[16] = "VOLUME_TSDF_RGB";

#define CU_VOXEL_MAX_W		0xffff // 65535
//#define CU_VOXEL_MAX_TSDF	0x7fff // 32767
#define CU_VOXEL_MAX_TSDF	0x7ffe // 32767

#define CU_VOXEL_UB_W		50.0f ///< upper bound of weight value
/*
/// RGB
typedef struct _CU_COLOR_T
{
	uchar	p[3];
} cu_color_t;

typedef struct _CU_VOXEL_T {
	short	tsdf;	// normalized tsdf
	ushort	w;		// weight
//	uchar3	color;	// color (BGR)
} cu_voxel_t;

typedef struct _CU_VOXEL {
	int3		size;
	int			size_xy;	// size.x * size.y
	float		min_t;		// min truncation e.g. 0.06 m
	float		max_t;		// max truncation e.g. 0.06 m
	float		grid_s;		// grid size e.g. 0.005 m
	cu_voxel_t	*data;		// volume data (device) pointer, [z][y][x]
} cu_voxel;
*/
#if 0
#define cu_voxel_access_z(v,_z) ((v)->data + (_z) * (v)->size_xy) /// [z][][]
#define cu_voxel_access_yz(v,_y,_z) (cu_voxel_access_z(v,_z) + (_y) * (v)->size.x) ///< [z][y][]
#define cu_voxel_access_xyz(v,_x,_y,_z) (cu_voxel_access_yz(v,_y,_z) + (_x)) ///< [z][y][x]

typedef struct _cu_kinect_fusion_params {
	float_camera_intrinsic	video_int[KINECT_VIDEO_LEVEL];
	float_camera_intrinsic	depth_int[KINECT_DEPTH_LEVEL];
	float_camera_extrinsic	depth_video_ext;	// depth to video transformation
	float_camera_extrinsic	video_depth_ext;	// video to depth transformation
	float_camera_extrinsic	global_ext;	// camera to global transformation
	float					base_line; // baseline (meter)
	float					disparity_offset; // disparity offset
	cu_voxel				volume;
	nwf_icp_projective_3d_params	icp_params;
	int	icp_it_max[KINECT_DEPTH_LEVEL];
	int2	video_size[KINECT_VIDEO_LEVEL];
	int2	depth_size[KINECT_VIDEO_LEVEL];
	uchar3	*h_video;	// raw video image
	short	*h_depth;	// raw depth image
	float	*h_disparity; // raw disparity image
	float2	*h_video_dmap;	// video image distortion map
	float2	*d_video_dmap;
	float2	*h_depth_dmap;	// depth image distortion map
	float2	*d_depth_dmap;
	float	*h_depth_to_meter;	// depth to meter table
	float	*d_depth_to_meter;
	// local variables
	size_t	video_size_t[KINECT_VIDEO_LEVEL];
	size_t	depth_size_t[KINECT_VIDEO_LEVEL];
	uchar3	*d_video;
	short	*d_depth;
	float	*d_disparity;
	uchar3	*d_video_u;	// undistorted video image
	short	*d_depth_u;	// undistorted depth image
	float	*d_disparity_u;	// undistorted disparity image
	float	*d_meter_x; // temporary meter buffer
	float3	*d_vertex_x; // temporary vertex buffer
	float	*d_meter_u[KINECT_VIDEO_LEVEL];	// undistorted depth image (meter)
	float3	*d_vertex_u[KINECT_VIDEO_LEVEL];
	float3	*d_normal_u[KINECT_VIDEO_LEVEL];
	float	*d_meter_s[KINECT_VIDEO_LEVEL];	// synthesized depth image (meter)
	float3	*d_vertex_s[KINECT_VIDEO_LEVEL];
	float3	*d_normal_s[KINECT_VIDEO_LEVEL];
	uchar3	*d_color_s[KINECT_VIDEO_LEVEL];
	uchar	*d_mask[2];
	uchar	*d_inlier;
#ifdef CG_KLT_TRACKER
	V3D_GPU::klt_tracker	klt;
#endif
	// execution time measurements
	uint	time_count, time_count_icp, time_count_of, time_count_surf, time_count_integration, time_count_extraction;
	INT64	time_buf;
	INT64	time_icp;
	INT64	time_of;
	INT64	time_surf;
	INT64	time_integration;
	INT64	time_extraction;
} cu_kinect_fusion_params;

inline cuda_function_header float_camera_intrinsic nw_to_cu_camera_intrinsic( nwf_cam_int_cv src )
{
	float_camera_intrinsic	dst;
	dst.fc = make_float2( src.fc[0], src.fc[1] );
	dst.cc = make_float2( src.cc[0], src.cc[1] );
	copy_float( dst.dc, src.dc, 5 );
	return dst;
}

inline cuda_function_header float_camera_extrinsic nw_to_cu_camera_extrinsic( nwf_cam_ext src )
{
	float_camera_extrinsic	dst;
	dst.R.p[0] = make_float3( src.R[0], src.R[1], src.R[2] );
	dst.R.p[1] = make_float3( src.R[3], src.R[4], src.R[5] );
	dst.R.p[2] = make_float3( src.R[6], src.R[7], src.R[8] );
	dst.t = make_float3( src.t[0], src.t[1], src.t[2] );
	return dst;
}

//////////////////////
// Device Functions //
//////////////////////

__device__ __forceinline__ short d_pack_tsdf( float tsdf_f )
{
	return /*__float2int_rz*/(int)( tsdf_f * CU_VOXEL_MAX_TSDF );
}

__device__ __forceinline__ float d_unpack_tsdf( short tsdf_i )
{
	return /*__int2float_rn*/(float)(tsdf_i) / CU_VOXEL_MAX_TSDF;
}

__device__ __forceinline__ ushort d_pack_weight( float w_f )
{
	w_f = (w_f < CU_VOXEL_UB_W) ? w_f / CU_VOXEL_UB_W : 1.0f;
	return /*__float2uint_rn*/(ushort)( w_f * CU_VOXEL_MAX_W + 0.5f );
}

__device__ __forceinline__ float d_unpack_weight( ushort w_i )
{
	return /*__int2float_rn*/(float)(w_i) / CU_VOXEL_MAX_W * CU_VOXEL_UB_W;
}

__global__ void d_float_data_to_mask( int size, float *d_data, uchar *d_mask );
__global__ void d_float_mask_to_data( int size, uchar *d_mask, float *d_data );
__global__ void d_kinect_depth_to_meter( int size, float *d_map, short *d_depth, float *d_meter );
__global__ void d_img_meter_to_vertex( float_camera_intrinsic k, int2 size, float *meter, float3 *vertex );
__global__ void d_img_vertex_to_normal( int2 size, float3 *vertex, float3 *normal );
__global__ void d_map_view_change( float34 P21, int2 size2, float3 *vertex2, float2 *map2 );
__global__ void d_uchar3_mapping( int2 src_size, int2 dst_size, float2 *d_map, uchar3 *d_src, uchar3 *d_dst );
__global__ void d_short_mapping( int2 src_size, int2 dst_size, float2 *d_map, short *d_src, short *d_dst );
__global__ void d_float_mapping( int2 src_size, int2 dst_size, float2 *d_map, float *d_src, float *d_dst );
__global__ void d_pyramid_down( int2 s_size, int2 d_size, float *d_s_ptr, uchar *d_s_m_ptr, 
	float *d_d_ptr, uchar *d_d_m_ptr, float sigma );

bool _cu_icp_projective_3d( nwf_icp_projective_3d_params icp, float_camera_intrinsic d2_int, float_camera_extrinsic *r_ext,
	int2 size1, int2 size2, float3 *d_vertex1, float3 *d_normal1, float3 *d_vertex2, float3 *d_normal2, uchar *d_inlier );
void _cu_voxel_volumetric_integration( cu_voxel v, float_camera_intrinsic ci, float_camera_extrinsic ce,
	int2 size, float *d_depth_ptr, float3 *d_normal_ptr, uchar3 *d_color_ptr, bool is_dynamic );
void _cu_voxel_extract_depth_and_normal_map( cu_voxel v, float_camera_intrinsic ci, float_camera_extrinsic ce,
	int2 size, float *d_ptr, float3 *n_ptr, uchar3 *c_ptr );

/////////////////
// Experiments //
/////////////////

__global__ void d_kinect_depth_to_meter( const ptr_step_size<short> depth_ptr, ptr_step<float> meter_ptr );
__global__ void d_float_meter_to_gradient( const ptr_step_size<float> meter_ptr, ptr_step<float2> gradient_ptr, int r, float sigma );
__global__ void d_float_camera_intrinsic_unproject_2( float_camera_intrinsic k, ptr_step_size<float2> image_ptr );
__global__ void d_float_camera_intrinsic_unproject_3( float_camera_intrinsic k, ptr_step_size<float> meter_ptr, ptr_step<float3> vertex_ptr );
__global__ void d_float_gradient_to_normal( const ptr_step_size<float> meter_ptr, const ptr_step<float2> gradient_ptr,
	const ptr_step<float2> image_ptr, ptr_step<float3> normal_ptr );
void cu_kinect_depth_to_vertex( float_camera_intrinsic k, int2 size, short *h_depth, float3 *h_vertex );
void d_float_meter_to_normal( float_camera_intrinsic k, const ptr_step_size<float> meter_ptr, ptr_step<float3> normal_ptr );
void cu_kinect_depth_to_normal( float_camera_intrinsic k, int2 size, short *h_depth, float3 *h_normal );

////////////////////
// Host Functions //
////////////////////

void cu_uchar3_normal_to_bgr( int size, float3 *d_normal, uchar3 *d_bgr );
void cu_meter_to_vertex( float_camera_intrinsic k, int2 size, float *h_meter, float3 *h_vertex );
void cu_map_view_change( float34 P21, int2 size2, float3 *vertex2, float2 *map2 );
void cu_pyramid_down( int2 s_size, float *h_s_ptr, unsigned char *h_s_m_ptr, float *h_d_ptr, unsigned char *h_d_m_ptr, float sigma );

void cu_voxel_new( cu_voxel *v, int3 size, float min_t, float max_t, float grid_s );
void cu_voxel_delete( cu_voxel *v );

void cu_voxel_run_length_encode( cu_voxel *v_local, char *file_name, int3 begin_local, int3 begin_global, int3 size );
void cu_voxel_run_length_decode( cu_voxel *v_local, char *file_name, int3 begin_local, int3 begin_global, int3 size );
void cu_voxel_rlc_to_point_cloud( char *file_input, char *file_output );
void cu_voxel_rlc_to_mesh( char *file_input, char *file_output );

void cu_kinect_fusion_new( cu_kinect_fusion_params *kf );
void cu_kinect_fusion_input( cu_kinect_fusion_params *kf );
void cu_kinect_fusion_render( cu_kinect_fusion_params *kf );
void cu_kinect_fusion_process( cu_kinect_fusion_params *kf, bool integration, bool tracking, bool extraction, bool is_dynamic );
void cu_kinect_fusion_delete( cu_kinect_fusion_params *kf );

#define KINECTFUSION_GRAP_FRAME     -2
#define KINECTFUSION_TRACKING_FAIL	-1
#define KINECTFUSION_TRACKING_NONE	0
#define KINECTFUSION_TRACKING_ICP	1
#define KINECTFUSION_TRACKING_OF	2
#define KINECTFUSION_TRACKING_SURF	3
#define KINECTFUSION_TRACKING_FU	4
int cu_kinect_fusion_process_multimodal( cu_kinect_fusion_params *kf, bool integration, int tracking, bool extraction );

void cu_kinect_depth_to_vertex( float_camera_intrinsic k, int2 size, short *h_depth, float3 *h_vertex );
void cu_kinect_depth_to_normal( float_camera_intrinsic k, int2 size, short *h_depth, float3 *h_normal );
void cu_image_disparity_to_normal( float_camera_intrinsic k, float base_line, float disparity_offset,
	int2 size, float *h_disparity, float3 *h_normal );
#endif
void cu_voxel_rlc_to_mesh( char *file_input, char *file_output , int volume_dim);

void cu_voxel_to_mesh(short2* _local_volume_data, unsigned int _size_x, unsigned int _size_y, unsigned int _size_z, float _grid_s, const char *file_output); //inyeop

__device__ __forceinline__ float d_unpack_weight( ushort w_i )
{
	return /*__int2float_rn*/(float)(w_i) / (float)16.0/*CU_VOXEL_MAX_W * CU_VOXEL_UB_W*/;
}


__device__ __forceinline__ float d_unpack_tsdf( short tsdf_i )
{
	return /*__int2float_rn*/(float)(tsdf_i) / CU_VOXEL_MAX_TSDF;
}

#endif

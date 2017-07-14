/**
\file		cu_kinectfusion_v1.cu
\brief		Cuda KIENCT FUSION library

\author		Seong-Oh Lee
\version	1.0
\date		2012.06.25
*/

//#include <cuda_runtime.h>
//#include <cuda.h>
//#include <device_launch_parameters.h>

// Helper functions

#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_math.h>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <fstream>
//#include <opencv2/nonfree/gpu.hpp>

#include "con_ui.h"
//#include "new_well/utility/file_io.h"
//#include "new_well/utility/utils.h"
//#include "new_well/bms/bms.h"
//#include "new_well/fill_data.h"
//#include "cu_bilateral_filter.h"
//#include "cu_nlm_filter.h"
#include "cu_kinect_fusion_v1.h"
#include "cu_memory.hpp"
//#include "new_well/algebra/mat.h"
#include <iostream>
#include <string.h>
#include <map>
#include "device_launch_parameters.h"


#if 0

/////////////////////////////
// CUDA libraries (global) //
/////////////////////////////

// normal map[-1,1] -> BGR[0,255]
__global__ void d_uchar3_normal_to_bgr( int size, float3 *d_normal, uchar3 *d_bgr )
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= size)
		return;
	float3 normal_ = d_normal[x];
	d_bgr[x] = make_uchar3( 127, 127, 127 );
	if( is_normal_valid(normal_) ) {
		const float		shift = 1.0f;
		const float		scale = 127.5f;
		d_bgr[x].z = (BYTE)(( normal_.x + shift ) * scale );
		d_bgr[x].y = (BYTE)(( shift - normal_.y ) * scale ); // inverse
		d_bgr[x].x = (BYTE)(( shift - normal_.z ) * scale ); // inverse
	}
}

__global__ void d_float_data_to_mask( int size, float *d_data, uchar *d_mask )
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= size)
		return;
	d_mask[x] = (d_data[x]) ? 0xff : 0;
}

__global__ void d_float_mask_to_data( int size, uchar *d_mask, float *d_data )
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= size)
		return;
	if( d_mask[x] == 0 ) d_data[x] = 0;
}

// fT	= focal length (pixel) * baseline (meter)
// o	= disparity offset
__global__ void d_float_disparity_to_meter( int size, float *d_disparity, float *d_meter, float fT, float o )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if( x >= size )
		return;
	set_meter_invalid( d_meter[x] );
	if( float1_non_zero( d_disparity[x] ) ) {
		d_meter[x] = fT / ( d_disparity[x] + o );
	}
}

/// \param size		[in] image size
/// \param d_map	[in] depth to meter map
/// \param d_depth	[in] depth image
/// \param d_meter	[out] meter image
__global__ void d_kinect_depth_to_meter( int size, float *d_map, short *d_depth, float *d_meter )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if( x >= size )
		return;
#if defined(DEVICE_OPENNI)
	d_meter[x] = (float)d_depth[x] / 1000.0f;
#elif defined(DEVICE_KINECT)
	d_meter[x] = (float)(d_depth[x]>>3) / 1000.0f;
#elif defined(DEVICE_FREENECT)
	d_meter[x] = (d_depth[x] < 2047) ? d_map[d_depth[x]] : 0.0f;
#endif
	// Limit valid range from 0.5 to 5 meter
	if( d_meter[x] < KINECT_DEPTH_MIN || d_meter[x] > KINECT_DEPTH_MAX ) d_meter[x] = 0.0f;
}

// Detph image to vertex image
__global__ void d_img_meter_to_vertex( float_camera_intrinsic k, int2 size, float *meter, float3 *vertex )
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	if( i>=size.y || j>=size.x )
		return;
	int x = i*size.x + j;
	set_vertex_invalid( vertex[x] );
	if( is_meter_valid(meter[x]) ) {
		vertex[x] = float_camera_intrinsic_unproject_3( k, make_float2( j, i ), meter[x] );
	}
}

// Vertex image to normal map
__global__ void d_img_vertex_to_normal( int2 size, float3 *vertex, float3 *normal )
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	
	if( i>=size.y || j>=size.x )
		return;
	float3	n_c; set_normal_invalid( n_c );
	int index = i*size.x + j;
	if( i < (size.y-1) && j < (size.x-1) ) {
		float3	v_c = vertex[index];
		float3	v_x = vertex[index+1];
		float3	v_y = vertex[index+size.x];
		if( is_vertex_valid(v_c) && is_vertex_valid(v_x) && is_vertex_valid(v_y) ) {
			float3	tx = float3_sub( v_x, v_c );
			float3	ty = float3_sub( v_c, v_y );
			n_c = float3_cross( tx, ty );
			n_c = float3_unit( n_c );
		}
	}
	normal[index] = n_c;
}

// Map 1st view to 2nd view
// P21 = float_camera_intrinsic (k1) x float_camera_extrinsic (T21)
__global__ void d_map_view_change( float34 P21, int2 size2, float3 *vertex2, float2 *map2 )
{
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	if( i>=size2.y || j>=size2.x )
		return;
	int index = i*size2.x + j;
	map2[index].x = FLT_MAX;
	if( is_vertex_valid( vertex2[index] ) ) {
		float3	x3 = float34_mv_3( P21, vertex2[index] );
		map2[index] = float2_make_inhomogeneous( x3 );
	}
}

/// \param src_size	[in] d_src size
/// \param dst_size	[in] d_map & d_dst size
/// \param d_map	[in] transformation map
/// \param d_src	[in] source image
/// \param d_dst	[out] destination image
__global__ void d_uchar3_mapping( int2 src_size, int2 dst_size, float2 *d_map, uchar3 *d_src, uchar3 *d_dst )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( y>=dst_size.y || x>=dst_size.x )
		return;
	int		index = y * dst_size.x + x;
	d_dst[index] = make_uchar3( 0, 0, 0 );
	int2	u = make_int2( __float2int_rn( d_map[index].x ), __float2int_rn( d_map[index].y ) );
	if( u.x<0 || u.y<0 || u.x>=src_size.x || u.y>=src_size.y )
		return;
	d_dst[index] = d_src[u.y * src_size.x + u.x];
}

/// \param src_size	[in] d_src size
/// \param dst_size	[in] d_map & d_dst size
/// \param d_map	[in] transformation map
/// \param d_src	[in] source image
/// \param d_dst	[out] destination image
__global__ void d_short_mapping( int2 src_size, int2 dst_size, float2 *d_map, short *d_src, short *d_dst )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( y>=dst_size.y || x>=dst_size.x )
		return;
	int		index = y * dst_size.x + x;
	d_dst[index] = 0;
	int2	u = make_int2( __float2int_rn( d_map[index].x ), __float2int_rn( d_map[index].y ) );
	if( u.x<0 || u.y<0 || u.x>=src_size.x || u.y>=src_size.y )
		return;
	d_dst[index] = d_src[u.y * src_size.x + u.x];
}

/// \param src_size	[in] d_src size
/// \param dst_size	[in] d_map & d_dst size
/// \param d_map	[in] transformation map
/// \param d_src	[in] source image
/// \param d_dst	[out] destination image
__global__ void d_float_mapping( int2 src_size, int2 dst_size, float2 *d_map, float *d_src, float *d_dst )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( y>=dst_size.y || x>=dst_size.x )
		return;
	int		index = y * dst_size.x + x;
	d_dst[index] = 0.f;
	int2	u = make_int2( __float2int_rn( d_map[index].x ), __float2int_rn( d_map[index].y ) );
	if( u.x<0 || u.y<0 || u.x>=src_size.x || u.y>=src_size.y )
		return;
	d_dst[index] = d_src[u.y * src_size.x + u.x];
}

__global__ void d_pyramid_down( int2 s_size, int2 d_size, float *d_s_ptr, uchar *d_s_m_ptr, 
	float *d_d_ptr, uchar *d_d_m_ptr, float sigma )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= d_size.x || y >= d_size.y )
		return;
	const int D = 5;
	const int D_half = (D>>1);
	int s_index = (y<<1) * s_size.x + (x<<1);
	int d_index = y * d_size.x + x;
	d_d_m_ptr[d_index] = 0x00;
	if( d_s_m_ptr[s_index] ) {
		float center = d_s_ptr[s_index];
		int tx = min ((x<<1) - D_half + D, s_size.x - 1);
		int ty = min ((y<<1) - D_half + D, s_size.y - 1);
		float sum = 0;
		int count = 0;
		for ( int cy = max (0, (y<<1) - D_half); cy < ty; ++cy) {
			for (int cx = max (0, (x<<1) - D_half); cx < tx; ++cx) {
				s_index = cy * s_size.x + cx;
				if( d_s_m_ptr[s_index] ) {
					float val = d_s_ptr[s_index];
					if( fabsf(val - center) < sigma ) {
						sum += val;
						++count;
					}
				}
			}
		}
		if( count ) {
			d_d_ptr[d_index] = sum / count;
			d_d_m_ptr[d_index] = 0xff;
		}
	}
}

///////////////////////
// General Functions //
///////////////////////

void cu_uchar3_normal_to_bgr( int size, float3 *d_normal, uchar3 *d_bgr )
{
	dim3 block = dim3( CU_BLOCK_MAX );
	dim3 grid = dim3( int_div_up(size, block.x) );
	d_uchar3_normal_to_bgr<<<grid,block>>>( size, d_normal, d_bgr );
	checkCudaErrors( cudaGetLastError() ); cu_sync();
}

void cu_meter_to_vertex( float_camera_intrinsic k, int2 size, float *h_meter, float3 *h_vertex )
{
	size_t		count = size.x * size.y;
	float		*d_meter;
	checkCudaErrors(cudaMalloc( &d_meter, count * sizeof(float) ));
	checkCudaErrors(cudaMemcpy( d_meter, h_meter, count * sizeof(float), cudaMemcpyHostToDevice ));
	float3		*d_vertex;
	checkCudaErrors(cudaMalloc( &d_vertex, count * sizeof(float3) ));

	dim3 block (CU_BLOCK_X, CU_BLOCK_Y);
	dim3 grid (int_div_up(size.x,block.x), int_div_up(size.y,block.y));
	d_img_meter_to_vertex<<<grid,block>>>( k, size, d_meter, d_vertex );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );

	checkCudaErrors(cudaMemcpy( h_vertex, d_vertex, count * sizeof(float3), cudaMemcpyDeviceToHost ));

	checkCudaErrors(cudaFree( d_meter ));
	checkCudaErrors(cudaFree( d_vertex ));
	checkCudaErrors( cudaGetLastError() ); cu_sync();
}

void cu_map_view_change( float34 P21, int2 size2, float3 *vertex2, float2 *map2 )
{
	size_t		size;
	float3		*d_vertex2;
	size = size2.x * size2.y *sizeof(float3);
	checkCudaErrors(cudaMalloc( &d_vertex2, size ));
	checkCudaErrors(cudaMemcpy( d_vertex2, vertex2, size, cudaMemcpyHostToDevice ));
	float2		*d_map2;
	size = size2.x * size2.y * sizeof(float2);
	checkCudaErrors(cudaMalloc( &d_map2, size ));

	dim3 block (CU_BLOCK_X, CU_BLOCK_Y);
	dim3 grid (int_div_up(size2.x,block.x), int_div_up(size2.y,block.y));
	d_map_view_change<<<grid,block>>>( P21, size2, d_vertex2, d_map2 );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );

	checkCudaErrors(cudaMemcpy( map2, d_map2, size, cudaMemcpyDeviceToHost ));

	checkCudaErrors(cudaFree( d_vertex2 ));
	checkCudaErrors(cudaFree( d_map2 ));
	checkCudaErrors( cudaGetLastError() ); cu_sync();
}

void cu_pyramid_down( int2 s_size, float *h_s_ptr, unsigned char *h_s_m_ptr, float *h_d_ptr, unsigned char *h_d_m_ptr, float sigma )
{
	int2 d_size = make_int2( (s_size.x)>>1, (s_size.y)>>1 );

	size_t	s_size_t = s_size.x * s_size.y;
	size_t	d_size_t = d_size.x * d_size.y;
	int size = s_size_t * sizeof(float);
	float	*d_s_ptr;
	checkCudaErrors( cudaMalloc( &d_s_ptr, size ) );
	checkCudaErrors(cudaMemcpy( d_s_ptr, h_s_ptr, size, cudaMemcpyHostToDevice));
	size = s_size_t * sizeof(unsigned char);
	unsigned char	*d_s_m_ptr;
	checkCudaErrors( cudaMalloc( &d_s_m_ptr, size ) );
	checkCudaErrors(cudaMemcpy( d_s_m_ptr, h_s_m_ptr, size, cudaMemcpyHostToDevice));

	size = d_size_t * sizeof(float);
	float	*d_d_ptr;
	checkCudaErrors( cudaMalloc( &d_d_ptr, size ) );
	size = d_size_t * sizeof(unsigned char);
	unsigned char	*d_d_m_ptr;
	checkCudaErrors( cudaMalloc( &d_d_m_ptr, size ) );

	dim3 blockSize(CU_BLOCK_X, CU_BLOCK_Y);
	dim3 gridSize( int_div_up(d_size.x,blockSize.x), int_div_up(d_size.y,blockSize.y) );
	d_pyramid_down<<< gridSize, blockSize>>>( s_size, d_size, d_s_ptr, d_s_m_ptr, d_d_ptr, d_d_m_ptr, sigma);

	size = d_size_t * sizeof(float);
	checkCudaErrors(cudaMemcpy( h_d_ptr, d_d_ptr, size, cudaMemcpyDeviceToHost));
	size = d_size_t * sizeof(unsigned char);
	checkCudaErrors(cudaMemcpy( h_d_m_ptr, d_d_m_ptr, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_s_ptr));
	checkCudaErrors(cudaFree(d_s_m_ptr));
	checkCudaErrors(cudaFree(d_d_ptr));
	checkCudaErrors(cudaFree(d_d_m_ptr));
	checkCudaErrors( cudaGetLastError() ); cu_sync();
}


//////////////////////
// Voxel Processing //
//////////////////////

__global__ void d_voxel_initialize( cu_voxel v )
{
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int z = threadIdx.y + blockIdx.y * blockDim.y;
	if( y>=v.size.y || z>=v.size.z )
		return;
	cu_voxel_t	*ptr_s = cu_voxel_access_yz( &v, y, z );
	cu_voxel_t	*ptr_e = ptr_s + v.size.x;
	do {
		ptr_s->tsdf	= -CU_VOXEL_MAX_TSDF;
		ptr_s->w	= 0;
		ptr_s->color.x = 0;
		ptr_s->color.y = 0;
		ptr_s->color.z = 0;
	} while( (++ptr_s) != ptr_e );
}

void cu_voxel_new( cu_voxel *v, int3 size, float min_t, float max_t, float grid_s )
{
	v->size = size;
	v->min_t	= min_t;
	v->max_t	= max_t;
	v->grid_s	= grid_s;
	v->size_xy = size.x * size.y;
	if( v->data == NULL)
	{
		
		checkCudaErrors(cudaMalloc( &v->data, v->size_xy * v->size.z * sizeof(cu_voxel_t) )); // FIXME
	}
	dim3 block (CU_BLOCK_X, CU_BLOCK_Y);
	dim3 grid (int_div_up(v->size.y,block.x), int_div_up(v->size.z,block.y));
	d_voxel_initialize<<<grid,block>>>( *v );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );
}


void cu_voxel_delete( cu_voxel *v )
{
	if( v->data ) {
		cu_free( v->data );
		v->data		= NULL;
		v->size.x	= 0;
		v->size.y	= 0;
		v->size.z	= 0;
		v->min_t	= 0.0f;
		v->max_t	= 0.0f;
		v->grid_s	= 0.0f;
		checkCudaErrors( cudaGetLastError() ); cu_sync();
	}
}

__global__ void d_voxel_volumetric_depth_scaling( float_camera_intrinsic d_int, int2 size, float *depth_ptr, float *depth_scaled_ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= size.x || y >= size.y )
		return;
	int		index = y * size.x + x;
	float2	p2 = float_camera_intrinsic_unproject_2( d_int, make_float2( x, y ) );
	depth_scaled_ptr[index] = depth_ptr[index] * sqrtf( float2_sum_sq(p2) + 1.0f );
}

__global__ void d_voxel_volumetric_integration( cu_voxel v,	float_camera_intrinsic ci, float_camera_extrinsic ce, int2 size,
	float *depth_scaled_ptr, float3 *normal_ptr, uchar3 *color_ptr, bool is_dynamic )
{
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int z = threadIdx.y + blockIdx.y * blockDim.y;
	if( y>=v.size.y || z>=v.size.z )
		return;
	float3	p3_curr = make_float3( -v.grid_s, y*v.grid_s, z*v.grid_s );
	float3	q3;
	float2	q2;
	int2	q2_i;
	float	sdf, sdf1, sdf2, weight, weight1, weight2, z_val, q3_norm;
	uchar3	color;
	cu_voxel_t	*v_ptr = cu_voxel_access_yz( &v, y, z );
	cu_voxel_t	voxel;
	for( int x=0; x<v.size.x; x++ ) {
		// convert current voxel from grid to global 3D position
		p3_curr.x += v.grid_s;
		q3 = float_camera_extrinsic_backward( ce, p3_curr );
		if( q3.z < /*KINECT_DEPTH_MIN*/FLT_MIN ) continue; // FIXME
		q2 = float_camera_intrinsic_project_3( ci, q3 );
		q2_i.x = __float2int_rn( q2.x ); q2_i.y = __float2int_rn( q2.y );
		if( q2_i.x<0 || q2_i.y<0 || q2_i.x>=(size.x-2) || q2_i.y>=(size.y-2) ) continue; // FIXME
		// current voxel is in camera view frustrum
		z_val = depth_scaled_ptr[q2_i.y*size.x + q2_i.x];
		if( z_val > FLT_MIN ) {
			q3_norm = float3_norm2(q3);
			sdf = z_val - q3_norm; // float3_distance(ce.t, p3_curr);
			if( sdf >= -v.min_t ) {
#if 1
				// Fixed truncation boundary
				if( sdf > 0.0f ) { // visible space into free space
					sdf /= v.max_t;
					if( sdf > 1.0f ) sdf = 1.0f;
				} else { // non-visible side
					sdf /= v.min_t;
					//if( sdf < -1.0f ) sdf = -1.0f;
				}				
#else
				// Adaptive truncation boundary based on the simplified noise model of Kinect
				// Reference: Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications
				float	max_t = 0.004275f * vector1_sq(z_val); // 0.004275 = 0.00285 * 1.5
				if( max_t < v.grid_s ) max_t = v.grid_s;
				if( sdf > 0.0f ) { // visible space into free space
					if( sdf > max_t*3.0f ) sdf = 1.0f;
					else	sdf /= v.max_t;
					if( sdf > 1.0f ) sdf = 1.0f;
				} else { // non-visible side
					if( sdf < -max_t*2.0f ) sdf = -1.0f;
					else sdf /= v.min_t;
				}
#endif
				// Note that the associated weight is proportional to cos(theta)/depth
#if 1
				weight = 1.0f;
				if( normal_ptr ) {
					// FIXME: unwanted noise added
					float3	p3 = normal_ptr[q2_i.y * size.x + q2_i.x];
					if( !is_normal_valid( p3 ) ) continue;
					weight = -float3_dot( p3, float3_div_c(q3, q3_norm) );
					//weight = 1.0f - fabs(float3_norm2(float3_cross(p3, q3)));
					if( weight < FLT_MIN ) continue;
				}
				weight /= z_val;
#else
				weight = 1.0f / z_val;
#endif
				// Read from global memory and unpack
				voxel = v_ptr[x];
				sdf1 = d_unpack_tsdf( voxel.tsdf );
				weight1 = d_unpack_weight( voxel.w );
				// Weighting, adapted based on a running average of the derivative of the TSDF
				if( is_dynamic ) {
					weight2 = fabsf( sdf - sdf1 );
					weight *= ( 1.0f + vector1_sq(weight2) * 10.0f );
				}
				// Update voxel
				//weight *= 10; // hwasup, faster adaptation
				weight2 = weight1 + weight;
				sdf2 = ( weight1*sdf1 + weight*sdf ) / weight2;
				// Update color
				color = color_ptr[q2_i.y * size.x + q2_i.x];
				// Check on satuaration (overexposed) and dark (underexposed)
#if 1
				if( (color.x>0 && color.x<255) || (color.y>0 && color.y<255) || (color.z>0 && color.z<255) ) {
					voxel.color.x = float1_to_uchar( ( weight1*voxel.color.x + weight*color.x ) / weight2 );
					voxel.color.y = float1_to_uchar( ( weight1*voxel.color.y + weight*color.y ) / weight2 );
					voxel.color.z = float1_to_uchar( ( weight1*voxel.color.z + weight*color.z ) / weight2 );
				}
#else
				voxel.color.x = float1_to_uchar( ( weight1*voxel.color.x + weight*color.x ) / weight2 );
				voxel.color.y = float1_to_uchar( ( weight1*voxel.color.y + weight*color.y ) / weight2 );
				voxel.color.z = float1_to_uchar( ( weight1*voxel.color.z + weight*color.z ) / weight2 );
#endif
				// Accelerate update (useless)
				//if( sdf2 > 0.999f ) sdf2 = 1.0f;
				//if( sdf2 < -0.999f ) sdf2 = -1.0f;

				// Pack and write to global memory
				voxel.tsdf = d_pack_tsdf( sdf2 );
				voxel.w = d_pack_weight( weight2 );
				v_ptr[x] = voxel;
			}
		}
	}
}

void _cu_voxel_volumetric_integration( cu_voxel v, float_camera_intrinsic ci, float_camera_extrinsic ce,
	int2 size, float *d_depth_ptr, float3 *d_normal_ptr, uchar3 *d_color_ptr, bool is_dynamic )
{
	dim3	block( CU_BLOCK_X, CU_BLOCK_Y );
	dim3	grid( int_div_up(size.x,block.x), int_div_up(size.y,block.y) );

	float	*d_meter_u_scaled;
	checkCudaErrors( cudaMalloc( &d_meter_u_scaled, size.x * size.y * sizeof(float) ));
	d_voxel_volumetric_depth_scaling<<<grid,block>>>( ci, size, d_depth_ptr, d_meter_u_scaled );
	cu_error(); cu_sync();
	grid = dim3( int_div_up(v.size.y,block.x), int_div_up(v.size.z,block.y) );
	d_voxel_volumetric_integration<<<grid,block>>>
		( v, ci, ce, size, d_meter_u_scaled, d_normal_ptr, d_color_ptr, is_dynamic );
	checkCudaErrors( cudaGetLastError() ); cu_sync();
	cu_free( d_meter_u_scaled );
}

__device__ float d_voxel_extract_trilinear_interpolated_sdf( cu_voxel *v, float3 p_f )
{
	int			sdf_i;
	int3		p_i;
	float3		t_f;
	float		sdfs[8];
	vector3_copy( p_i, p_f );
	vector3_sub( t_f, p_f, p_i );
	
	sdf_i = cu_voxel_access_xyz( v, p_i.x  , p_i.y  , p_i.z   )->tsdf;	// V000
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF\n");
		return -1.0f;
	}
	sdfs[0] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x+1, p_i.y  , p_i.z   )->tsdf;	// V100
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 1\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 1\n");
		return -1.0f;
	}
	sdfs[1] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x  , p_i.y+1, p_i.z   )->tsdf;	// V010
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 2\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 2\n");
		return -1.0f;
	}
	sdfs[2] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x  , p_i.y  , p_i.z+1 )->tsdf;	// V001
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 3\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 3\n");
		return -1.0f;
	}
	sdfs[3] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x+1, p_i.y  , p_i.z+1 )->tsdf;	// V101
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 4\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 4\n");
		return -1.0f;
	}
	sdfs[4] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x  , p_i.y+1, p_i.z+1 )->tsdf;	// V011
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 5\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 5\n");
		return -1.0f;
	}
	sdfs[5] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x+1, p_i.y+1, p_i.z   )->tsdf;	// V110
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 6\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 6\n");
		return -1.0f;
	}
	sdfs[6] = d_unpack_tsdf( sdf_i );
	sdf_i = cu_voxel_access_xyz( v, p_i.x+1, p_i.y+1, p_i.z+1 )->tsdf;	// V111
	if( sdf_i == CU_VOXEL_MAX_TSDF )
	{
		//printf("CU_VOXEL_MAX_TSDF 7\n");
		return 1.0f;
	}
	if( sdf_i == -CU_VOXEL_MAX_TSDF )
	{
		//printf("-CU_VOXEL_MAX_TSDF 7\n");
		return -1.0f;
	}
	//if( sdf_i == CU_VOXEL_MAX_TSDF ) return 1.0f;
	//if( sdf_i == -CU_VOXEL_MAX_TSDF ) return -1.0f;
	sdfs[7] = d_unpack_tsdf( sdf_i );
	return float_interpolation_trilinear( sdfs, t_f.x, t_f.y, t_f.z );
}

__device__ uchar3 _d_voxel_extract_surface_color( int width, cu_voxel_t *data1_ptr, cu_voxel_t *data2_ptr, float3 p_f, int3 p_i )
{
	cu_voxel_t	*data_ptr;
	float3		t_f;
	vector3_sub( t_f, p_f, p_i );
	int	index = p_i.y * width + p_i.x;

	cu_voxel_t	vox[8];
	data_ptr = data1_ptr + index;	vox[0] = *data_ptr;
	data_ptr += 1;					vox[1] = *data_ptr;
	data_ptr += width;				vox[6] = *data_ptr;
	data_ptr -= 1;					vox[2] = *data_ptr;
	data_ptr = data2_ptr + index;	vox[3] = *data_ptr;
	data_ptr += 1;					vox[4] = *data_ptr;
	data_ptr += width;				vox[7] = *data_ptr;
	data_ptr -= 1;					vox[5] = *data_ptr;

	float		vals[8];
	uchar3		color;
	for( int i=0; i<8; i++ ) vals[i] = (float)vox[i].color.x;
	color.x = float1_to_uchar( float_interpolation_trilinear( vals, t_f.x, t_f.y, t_f.z ) );
	for( int i=0; i<8; i++ ) vals[i] = (float)vox[i].color.y;
	color.y = float1_to_uchar( float_interpolation_trilinear( vals, t_f.x, t_f.y, t_f.z ) );
	for( int i=0; i<8; i++ ) vals[i] = (float)vox[i].color.z;
	color.z = float1_to_uchar( float_interpolation_trilinear( vals, t_f.x, t_f.y, t_f.z ) );
	return color;
}

__device__ uchar3 d_voxel_extract_surface_color( cu_voxel *v, float3 p_f )
{
	int3 p_i;
	vector3_copy( p_i, p_f );
	cu_voxel_t	*data_ptr = cu_voxel_access_z( v, p_i.z );
	return _d_voxel_extract_surface_color( v->size.x, data_ptr, data_ptr + v->size_xy, p_f, p_i );
}

__device__ float3 _d_voxel_extract_surface_normal( int width, cu_voxel_t *data1_ptr, cu_voxel_t *data2_ptr, float3 p_f, int3 p_i )
{
	cu_voxel_t	*data_ptr;
	short		sdf_i;
	float3		t_f, n_f;
	float		sdfs[8], vxy[4], sdf1, sdf2;
	int	index = p_i.y * width + p_i.x;
	vector3_set( n_f, 0.0f );
	vector3_sub( t_f, p_f, p_i );

	data_ptr = data1_ptr + index;
	sdf_i = data_ptr->tsdf;	// V000
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[0] = d_unpack_tsdf( sdf_i );

	data_ptr += 1;
	sdf_i = data_ptr->tsdf;	// V100
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[1] = d_unpack_tsdf( sdf_i );

	data_ptr += width;
	sdf_i = data_ptr->tsdf;	// V110
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[6] = d_unpack_tsdf( sdf_i );

	data_ptr -= 1;
	sdf_i = data_ptr->tsdf;	// V010
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[2] = d_unpack_tsdf( sdf_i );

	data_ptr = data2_ptr + index;
	sdf_i = data_ptr->tsdf;	// V001
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[3] = d_unpack_tsdf( sdf_i );

	data_ptr += 1;
	sdf_i = data_ptr->tsdf;	// V101
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[4] = d_unpack_tsdf( sdf_i );

	data_ptr += width;
	sdf_i = data_ptr->tsdf;	// V111
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[7] = d_unpack_tsdf( sdf_i );

	data_ptr -= 1;
	sdf_i = data_ptr->tsdf;	// V011
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[5] = d_unpack_tsdf( sdf_i );

#if 1
	// x component
	vxy[0] = sdfs[0]; vxy[1] = sdfs[2]; vxy[2] = sdfs[3]; vxy[3] = sdfs[5];
	sdf1 = float_interpolation_bilinear( vxy, t_f.y, t_f.z );
	vxy[0] = sdfs[1]; vxy[1] = sdfs[6]; vxy[2] = sdfs[4]; vxy[3] = sdfs[7];
	sdf2 = float_interpolation_bilinear( vxy, t_f.y, t_f.z );
	n_f.x = sdf2 - sdf1;
	// y component
	vxy[0] = sdfs[0]; vxy[1] = sdfs[1]; vxy[2] = sdfs[3]; vxy[3] = sdfs[4];
	sdf1 = float_interpolation_bilinear( vxy, t_f.x, t_f.z );
	vxy[0] = sdfs[2]; vxy[1] = sdfs[6]; vxy[2] = sdfs[5]; vxy[3] = sdfs[7];
	sdf2 = float_interpolation_bilinear( vxy, t_f.x, t_f.z );
	n_f.y = sdf2 - sdf1;
	// z component
	vxy[0] = sdfs[0]; vxy[1] = sdfs[1]; vxy[2] = sdfs[2]; vxy[3] = sdfs[6];
	sdf1 = float_interpolation_bilinear( vxy, t_f.x, t_f.y );
	vxy[0] = sdfs[3]; vxy[1] = sdfs[4]; vxy[2] = sdfs[5]; vxy[3] = sdfs[7];
	sdf2 = float_interpolation_bilinear( vxy, t_f.x, t_f.y );
	n_f.z = sdf2 - sdf1;
#else
	// Significantly degrade the performance!
	// The piecewise constant gradient (using a constant average of finite differences discretization)
	// Reference: Eq.11 of F. Calakli and G. Taubin, "Smooth Signed Distance Surface Reconstruction"
	n_f.x = sdfs[1] - sdfs[0] + sdfs[4] - sdfs[3] + sdfs[6] - sdfs[2] + sdfs[7] - sdfs[5];
	n_f.y = sdfs[2] - sdfs[0] + sdfs[5] - sdfs[3] + sdfs[6] - sdfs[1] + sdfs[7] - sdfs[4];
	n_f.z = sdfs[3] - sdfs[0] + sdfs[5] - sdfs[2] + sdfs[4] - sdfs[1] + sdfs[7] - sdfs[6];
//	return float3_div_c( n_f, 4.0 * 0.003f ); // FIXME
#endif
	return float3_unit( n_f );
}

__device__ float3 d_voxel_extract_surface_normal( cu_voxel *v, float3 p_f )
{
	int3 p_i;
	vector3_copy( p_i, p_f );
	cu_voxel_t	*data_ptr = cu_voxel_access_z( v, p_i.z );
	return _d_voxel_extract_surface_normal( v->size.x, data_ptr, data_ptr + v->size_xy, p_f, p_i );
}

__global__ void d_voxel_extract_depth_and_normal_map( cu_voxel v, float_camera_intrinsic ci, float_camera_extrinsic ce,
	int2 size, float *d_ptr, float3 *n_ptr, uchar3 *c_ptr, float33 IM, float3 camera_center )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= size.x || y>=size.y )
		return;
	int		index = y * size.x + x;
	float	t, t1, t2, sdf, sdf1 = 1.0f; // set to free space
	float3	ray_dir, t_pos, n_pos;
	float2	vec2_1, p2;
	const float margin = 1;
	float	x_max = (float)v.size.x - margin;
	float	y_max = (float)v.size.y - margin;
	float	z_max = (float)v.size.z - margin;
	const float	dt_max = 1.0f; // 1.0f is a good choice.
	const float	dt_min = 0.25f; // 0.25f is a good choice.
	float	dt = dt_max;
	bool	dt_status = false;
	float	t_min, t_max;
	// set default
	
	d_ptr[index] = 0.f;
	n_ptr[index] = make_float3( 0.f, 0.f, 0.f );
	c_ptr[index] = make_uchar3( 255, 255, 255 );
	// get a ray direction and convert to grid position
	vec2_1.x = (float)x; vec2_1.y = (float)y;

	ray_dir = float33_mv_2( IM, vec2_1 ); // ray_dir's norm is approximately 1.0f.
	ray_dir = float3_unit( ray_dir ); // FIXME	
	
	//printf("ray_dir %f %f %f\n",ray_dir.x, ray_dir.y, ray_dir.z);

	//ensure that it isn't a degenerate case
	if(ray_dir.x == 0.f) ray_dir.x = 1e-15;
	if(ray_dir.y == 0.f) ray_dir.y = 1e-15;
	if(ray_dir.z == 0.f) ray_dir.z = 1e-15;

	// set t range = [t_min,t_max]
	t_min = FLT_MAX;
	t_max = 0;
	// x = 0
	t	= ( margin - camera_center.x ) / ray_dir.x;
	if( t > 0 ) {
		//printf("x=0\n");
		t1	= ray_dir.y * t + camera_center.y;
		t2	= ray_dir.z * t + camera_center.z;
		
		//printf("x=0 %f %f %f %f\n", t1, t2, y_max, z_max);
	
		if( t1 > margin && t1 < y_max && t2 > margin && t2 < z_max ) {
			if( t < t_min ) t_min = t;
			if( t > t_max ) t_max = t;
		}
	}
	// x = x_max
	t	= ( x_max - camera_center.x ) / ray_dir.x;
	if( t > 0 ) {
		//printf("x= x_max\n");
		t1	= ray_dir.y * t + camera_center.y;
		t2	= ray_dir.z * t + camera_center.z;
	
		//printf("x= x_max %f %f %f %f\n", t1, t2, y_max, z_max);
	
		if( t1 > margin && t1 < y_max && t2 > margin && t2 < z_max ) {
			if( t < t_min ) t_min = t;
			if( t > t_max ) t_max = t;
		}
	}
	// y = 0
	t	= ( margin - camera_center.y ) / ray_dir.y;
	if( t > 0 ) {
		t1	= ray_dir.x * t + camera_center.x;
		t2	= ray_dir.z * t + camera_center.z;
		
		//printf("y=0 %f %f %f %f\n", t1, t2, x_max, z_max);
	
		if( t1 > margin && t1 < x_max && t2 > margin && t2 < z_max ) {
			if( t < t_min ) t_min = t;
			if( t > t_max ) t_max = t;
		}
	}
	// y = y_max
	t	= ( y_max - camera_center.y ) / ray_dir.y;
	if( t > 0 ) {
		//printf("y= y_max\n");
		t1	= ray_dir.x * t + camera_center.x;
		t2	= ray_dir.z * t + camera_center.z;
		//printf("y=y_max %f %f %f %f\n", t1, t2, x_max, z_max);
	
		if( t1 > margin && t1 < x_max && t2 > margin && t2 < z_max ) {
			if( t < t_min ) t_min = t;
			if( t > t_max ) t_max = t;
		}
	}
	// z = 0
	t	= ( margin - camera_center.z ) / ray_dir.z;
	if( t > 0 ) {
		//printf("z=0\n");
		t1	= ray_dir.x * t + camera_center.x;
		t2	= ray_dir.y * t + camera_center.y;
	
		//printf("z=0 %f %f %f %f\n", t1, t2, x_max, y_max);
	
		if( t1 > margin && t1 < x_max && t2 > margin && t2 < y_max ) {
			if( t < t_min ) t_min = t;
			if( t > t_max ) t_max = t;
		}
	}
	// z = z_max
	t	= ( z_max - camera_center.z ) / ray_dir.z;
	if( t > 0 ) {
		//printf("z= z_max\n");
		t1	= ray_dir.x * t + camera_center.x;
		t2	= ray_dir.y * t + camera_center.y;

		//printf("z= z_max %f %f %f %f\n", t1, t2, x_max, y_max);
	
		if( t1 > margin && t1 < x_max && t2 > margin && t2 < y_max ) {
			if( t < t_min ) t_min = t;
			if( t > t_max ) t_max = t;
		}
	}
	if( t_min == FLT_MAX && t_max == 0 )
	{
		return;
	}
	if( t_min == t_max ) t_min = dt * margin;
	// scan voxels along ray
	for( t=t_min; t<t_max; t+=dt ) {
		t_pos = float3_mul_c( ray_dir, t );
		t_pos = float3_add( t_pos, camera_center );
		// boundary check: FIXME
		if( t_pos.x<0.0f || t_pos.y<0.0f || t_pos.z<0.0f ||
			t_pos.x>=x_max || t_pos.y>=y_max || t_pos.z>=z_max )
		{
			continue;
		}
		//sdf = d_unpack_tsdf( cu_voxel_access_xyz( &v, (int)t_pos.x, (int)t_pos.y, (int)t_pos.z )->tsdf );
		sdf = d_voxel_extract_trilinear_interpolated_sdf( &v, t_pos );
		// zero crossing check
		//printf("2 sdf: %f sdf1: %f\n",sdf,sdf1);
			
		if( sdf<0.0f && sdf>-1.0f && sdf1>=0.0f && sdf1<1.0f ) {
		
			//sdf = d_voxel_extract_trilinear_interpolated_sdf( &v, t_pos );
			//t_pos = float3_mul_c( ray_dir, t1 );
			//t_pos = float3_add( t_pos, camera_center );
			//sdf1 = d_voxel_extract_trilinear_interpolated_sdf( &v, t_pos );

			t2 = t1 - dt * sdf1 / (sdf - sdf1 ); // eq.(15)
			n_pos = float3_mul_c( ray_dir, t2 );
			t_pos = float3_add( n_pos, camera_center );
			// depth	
			p2 = float_camera_intrinsic_unproject_2( ci, make_float2( x, y ) );
			d_ptr[index] = float3_norm2(n_pos) / sqrtf( float2_sum_sq(p2) + 1.0f ) * v.grid_s;
			// normal
			n_pos = d_voxel_extract_surface_normal( &v, t_pos );
			n_ptr[index] = float33_tv( ce.R, n_pos );
			// color
			c_ptr[index] = d_voxel_extract_surface_color( &v, t_pos );
			break;
		}
		else {
			// back face check
			if( sdf1<=0.0f && sdf1>-1.0f && sdf>=0.0f && sdf<1.0f )
				break;
		}
		// update
		if( sdf > -1.0f && sdf < 1.0f )	{
			if( dt_status == false ) {
				t -= dt;
				dt_status = true;
				dt = dt_min;
				continue;
			}
		}
		else if( dt_status == true ) {
			dt_status = false;
			dt = dt_max;
		}
		t1 = t;
		sdf1 = sdf;
	}
}

void _cu_voxel_extract_depth_and_normal_map( cu_voxel v, float_camera_intrinsic ci, float_camera_extrinsic ce,
	int2 size, float *d_ptr, float3 *n_ptr, uchar3 *c_ptr )
{
	float33	K = float_camera_intrinsic_to_K( ci );
	float33	M, IM;
	M = float33_mt( K, ce.R );
	if( float33_inv( &IM, M ) ) {
		float3	camera_center = float3_div_c( ce.t, v.grid_s );
		dim3	block( CU_BLOCK_X, CU_BLOCK_Y );
		dim3	grid( int_div_up(size.x,block.x), int_div_up(size.y,block.y));
		d_voxel_extract_depth_and_normal_map<<<grid,block>>>( v, ci, ce, size, d_ptr, n_ptr, c_ptr, IM, camera_center );
		checkCudaErrors( cudaGetLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
	}
}

__global__ void d_icp_projective_3d_step1( float_camera_intrinsic camera2, float_camera_extrinsic pose_rel, nwf_icp_projective_3d_params param,
	int2 size1, int2 size2, float3 *d_v1, float3 *d_n1, float3 *d_v2, float3 *d_n2, float3 *d_v21 )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= size1.x || y >= size1.y )
		return;
	int	index = y * size1.x + x;
	float3	_d_v1	= d_v1[index];
	float3	_d_n1	= d_n1[index];
	float3	*_d_v21	= (d_v21 + index);
	float3	p3;
	float2	p2;
	int2	p2_i;
	// Find a corresponding point
	_d_v21->x = FLT_MAX;
	if( _d_n1.x || _d_n1.y || _d_n1.z ) {
		p3 = float_camera_extrinsic_backward( pose_rel, _d_v1 );
		//		if( p3.z > FLT_MIN ) {
		p2 = float_camera_intrinsic_project_3( camera2, p3 );
		p2_i.x = __float2int_rn( p2.x ); p2_i.y = __float2int_rn( p2.y );
		if( p2_i.x>=0 && p2_i.y>=0 && p2_i.x<size2.x && p2_i.y<size2.y ) {
			// Test a corresponding point
			index = p2_i.y*size2.x + p2_i.x;
			float3	_d_n2 = d_n2[index];
			if( _d_n2.x || _d_n2.y || _d_n2.z ) {
		
				_d_n2 = float33_mv( pose_rel.R, _d_n2 );
				//if( float3_norm2(float3_cross(p3, _d_n2)) < param.nt ) { // FIXME
				if( float3_dot(_d_n1, _d_n2) > param.nt ) {
		
					float3	_d_v2 = d_v2[index];
					_d_v2 = float_camera_extrinsic_forward( pose_rel, _d_v2 );
					if( float3_distance(_d_v1, _d_v2) < param.dt ) {
						*_d_v21 = _d_v2;
					}
				}
			}
		}
		//		}
	}
}

__global__ void d_icp_projective_3d_step4( float_camera_intrinsic camera2, float_camera_extrinsic pose_rel, nwf_icp_projective_3d_params param,
	int2 size1, int2 size2, float3 *d_v1, float3 *d_n1, float3 *d_v2, float3 *d_n2, float3 *d_v21 )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= size1.x || y >= size1.y )
		return;
	int	index = y * size1.x + x;
	float3	_d_v1	= d_v1[index];
	float3	_d_n1	= d_n1[index];
	float3	*_d_v21	= (d_v21 + index);
	float3	p3;
	float2	p2;
	int2	p2_i;
	// Find a corresponding point
	_d_v21->x = FLT_MAX;
	if( _d_n1.x || _d_n1.y || _d_n1.z ) {
		p3 = float_camera_extrinsic_backward( pose_rel, _d_v1 );
		p2 = float_camera_intrinsic_project_3( camera2, p3 );
		p2_i.x = (int)( p2.x ); p2_i.y = (int)( p2.y );
		if( p2_i.x>=0 && p2_i.y>=0 && p2_i.x<(size2.x-1) && p2_i.y<(size2.y-1) ) {
			// Test a corresponding point (sub-pixel correnspodence using the bilinear interpolation)
			index = p2_i.y*size2.x + p2_i.x;
			float3	nn_[4];
			nn_[0] = d_n2[index]; nn_[1] = d_n2[index+1]; nn_[2] = d_n2[index+size2.x]; nn_[3] = d_n2[index+size2.x+1];
			if( is_normal_valid(nn_[0]) && is_normal_valid(nn_[1]) && is_normal_valid(nn_[2]) && is_normal_valid(nn_[3]) ) {
				float	du_ = p2.x - p2_i.x, dv_ = p2.y - p2_i.y;
				float3	_d_n2 = vector3_interpolation_bilinear( nn_, du_, dv_ );
				_d_n2 = float33_mv( pose_rel.R, _d_n2 );
				if( float3_dot(_d_n1, _d_n2) > param.nt ) {
					nn_[0] = d_v2[index]; nn_[1] = d_v2[index+1]; nn_[2] = d_v2[index+size2.x]; nn_[3] = d_v2[index+size2.x+1];
					float3	_d_v2 = vector3_interpolation_bilinear( nn_, du_, dv_ );
					_d_v2 = float_camera_extrinsic_forward( pose_rel, _d_v2 );
					if( float3_distance(_d_v1, _d_v2) < param.dt ) {
						*_d_v21 = _d_v2;
					}
				}
			}
		}
	}
}

__global__ void d_icp_projective_3d_step2( int2 size1, float3 *d_v1, float3 *d_n1, float3 *d_v21, float *d_A, float *d_b )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= size1.x || y >= size1.y )
		return;
	int		index = y*size1.x + x;
	int		rows = size1.x * size1.y;
	float3	_d_v21	= d_v21[index];
	if( _d_v21.x == FLT_MAX ) {
		d_A += index;	*d_A = 0;
		d_A += rows;	*d_A = 0;
		d_A += rows;	*d_A = 0;
		d_A += rows;	*d_A = 0;
		d_A += rows;	*d_A = 0;
		d_A += rows;	*d_A = 0;
		d_b[index]	= 0;
	}
	else {
		
		float3	_d_v1	= d_v1[index];
		float3	_d_n1	= d_n1[index];
		d_A += index;	*d_A = _d_n1.z*_d_v21.y - _d_n1.y*_d_v21.z;
		d_A += rows;	*d_A = _d_n1.x*_d_v21.z - _d_n1.z*_d_v21.x;
		d_A += rows;	*d_A = _d_n1.y*_d_v21.x - _d_n1.x*_d_v21.y;
		d_A += rows;	*d_A = _d_n1.x;
		d_A += rows;	*d_A = _d_n1.y;
		d_A += rows;	*d_A = _d_n1.z;
		d_b[index] = ( _d_n1.x*_d_v1.x + _d_n1.y*_d_v1.y + _d_n1.z*_d_v1.z -
			_d_n1.x*_d_v21.x - _d_n1.y*_d_v21.y - _d_n1.z*_d_v21.z );
	}
}

__global__ void d_icp_projective_3d_step3( int size, float3 *d_v21, uchar *d_inlier )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if( x >= size )
		return;
	d_inlier[x] = (d_v21[x].x == FLT_MAX) ? 0 : 0xff;
}

/// We assume that the two vertex maps are captured from same camera.
/// Note that only an incremental transformation occurs
/// T(R,t) = 2 to 1 transformation
/// \param ci		[in] 2nd camera intrinsic parameters
/// \param ce		[in, out] camera extrinsic parameters
/// \param vm_1		[in] vertex map 1
/// \param nm_1		[in] normalized normal map 1
/// \param vm_2		[in] vertex map 2
/// \param nm_2		[in] normalized normal map 2
/// \param param	[in] ICP parameters
/// \param im		[out] an inlier map
/// \return			# of correspondences
bool _cu_icp_projective_3d( nwf_icp_projective_3d_params icp, float_camera_intrinsic d2_int, float_camera_extrinsic *r_ext,
	int2 size1, int2 size2, float3 *d_vertex1, float3 *d_normal1, float3 *d_vertex2, float3 *d_normal2, uchar *d_inlier )
{
	float_camera_extrinsic	pose_rel = *r_ext;
	// global memory allocation and copy (device)
	int		v1_size = size1.x * size1.y;
	size_t	size;
	float3	*d_v21;
	size = v1_size * sizeof(float3);
	checkCudaErrors(cudaMalloc( &d_v21, size ));
	float	*d_A;
	size = v1_size * 6 * sizeof(float);
	checkCudaErrors(cudaMalloc( &d_A, size ));
	float	*d_b;
	size = v1_size * sizeof(float);
	checkCudaErrors(cudaMalloc( &d_b, size ));
	float	*d_t;
	size = v1_size * sizeof(float);
	checkCudaErrors(cudaMalloc( &d_t, size ));
	thrust::device_ptr<float> _d_t( d_t );
	float	*d_AtA;
	size = 6 * 6 * sizeof(float);
	checkCudaErrors(cudaMalloc( &d_AtA, size ));
	float	*d_Atb;
	size = 6 * sizeof(float);
	checkCudaErrors(cudaMalloc( &d_Atb, size ));

	dim3	block (CU_BLOCK_X, CU_BLOCK_Y);
	dim3	grid (int_div_up(size1.x,block.x), int_div_up(size1.y,block.y));
	bool	rt = true; // return value
	int		it = icp.it_max;
	float	md = FLT_MAX;
	do {
		d_icp_projective_3d_step1<<<grid,block>>>
			( d2_int, pose_rel, icp, size1, size2, d_vertex1, d_normal1, d_vertex2, d_normal2, d_v21 );
		cu_error();	cu_sync();
		// Check the stopping conditions
		// Estimate transformation matrix
		d_icp_projective_3d_step2<<<grid,block>>>( size1, d_vertex1, d_normal1, d_v21, d_A, d_b );
		cu_error();	cu_sync();
		for( int i=0; i<6; i++ ) {
			for( int j=i; j<6; j++ ) {
				nppsMul_32f( d_A+v1_size*i, d_A+v1_size*j, d_t, v1_size );
				thrust::inclusive_scan( _d_t, _d_t+v1_size, _d_t );
				nppsCopy_32f( d_t+v1_size-1, d_AtA+j*6+i, 1 );
			}
			nppsMul_32f( d_A+v1_size*i, d_b, d_t, v1_size );
			thrust::inclusive_scan( _d_t, _d_t+v1_size, _d_t );
			nppsCopy_32f( d_t+v1_size-1, d_Atb+i, 1 );
		}

		// Note that covariance matrix C=ATA determines the change in error when surfaces are moved from optimal alignment
		// We assume that A has full rank 6 (AtA is symmetric positive definite).
		nwf_mat	*AtA = nwf_mat_new( 6, 6, FALSE );
		nwf_vec	*Atb = nwf_vec_new( 6 );
		cu_memcpy_device_to_host( d_AtA, AtA->p, 6 * 6 );
		cu_memcpy_device_to_host( d_Atb, Atb->p, 6 );
		// symmetry
		int count = 0;
		for( int i=0; i<6; i++ ) {
			for( int j=0; j<i; j++ ) {
				nw_mat_c( AtA, i, j ) = nw_mat_c( AtA, j, i );
				if( nw_mat_c( AtA, i, j ) ) count++;
			}
		}
		if( count == 0 ) rt = false; // empty check
		// The stability and validity checks
		// check on the null space
		//if( rt ) {
			//// To determine a unique solution
			//for( int i=0; i<6; i++ ) {
			//	nw_mat_c( AtA, i, i ) += 1e-5;
			//}
		//if( fabsf(nwf_mat_det( AtA )) < 1e-15 ) { // FIXME
		//	nw_cui_error( "cu_icp_projective_3d - null space." );
		//	rt = false;
		//}
		//}
		// The condition number of A is k(A) = ||A||·||A -1|| (norm(A)·norm(inv(A))).
		// If k(A) for least squares is large, that means:
		// 1. Small relative changes to A can cause large changes to the span of A
		//    (i.e. there are some vectors in the span of A^ that form a large angle with all the vectors in the span of A).
		// 2. The linear system to nd x in terms of the projection onto A will be ill-conditioned.
		// The lapack function fo estimating the reciprocal of the condition number is spocon_().
		if( rt ) {
			nwf_cho	*cho = nwf_cho_new( AtA );
			rt = false;
			// Check on the null space
			if( cho && fabsf(nwf_cho_det( cho ))>1e-15 ) {
				// Check on the condition number
				float anorm = nwf_lapack_lansy( '1', 'L', AtA->p, AtA->cols ); // lapack problem here! fix me!! 
				float rcond = nwf_cho_rcond( cho, anorm ); // lapack problem here! fix me!!
#if 1 // error check
				if( rcond > 5e-5 ) // lapack problem here! fix me!
		{
					nwf_vec	*_x = nwf_cho_back_sub_v( cho, Atb );
					float	_md = _nwf_vec_dot( Atb->p, _x->p, Atb->dim ); // mahalanobis distance
					nwf_cho_delete( cho );
					// Check on the convergence
					//if(	_md > md ) {
					//	nw_cui_error( "cu_icp_projective_3d - diverge" );
					//}
					// Check on the magnitude of incremental transform parameters
					if( fabsf(_x->p[0]) > icp.rt || fabsf(_x->p[1]) > icp.rt || fabsf(_x->p[2]) > icp.rt || 
						fabsf(_x->p[3]) > icp.tt || fabsf(_x->p[4]) > icp.tt || fabsf(_x->p[5]) > icp.tt ) {
							//				nwf_vec3_2nd_norm( _x->p + 3 ) > param.tt ) {
							nw_cui_error( "cu_icp_projective_3d - out of bound." );
					}
					else {
						// Update transformation matrix
						float_camera_extrinsic	pose_inc;
						nwf_SO3_from_so3( (float*)pose_inc.R.p, _x->p );
						pose_inc.t = make_float3( _x->p[3], _x->p[4], _x->p[5] );
						pose_rel = float_camera_extrinsic_compose( pose_rel, pose_inc );
						// Check on the convergence
						if( _md > md || _md < 1e-5 ) it = 1;
						md = _md;
						rt = true;
#if 0
						bool converged = true;
						for( int i=0; i<6; i++ ) {
							if( _x->p[i] > 1e-4 ) { // 0.0057 degree and 0.1 mm
								converged = false;
								break;
							}
						}
						if( converged ) {
							//nw_cui_debug( "cu_icp_projective_3d - %d converged", it );
							it = 1;
						}
#endif
					}
					nwf_vec_delete( _x );
				}
#else // for experiment only
				nwf_vec	*_x = nwf_cho_back_sub_v( cho, Atb );
				float	_md = _nwf_vec_dot( Atb->p, _x->p, Atb->dim ); // mahalanobis distance
				nwf_cho_delete( cho );
				// Update transformation matrix
				float_camera_extrinsic	pose_inc;
				nwf_SO3_from_so3( (float*)pose_inc.R.p, _x->p );
				pose_inc.t = make_float3( _x->p[3], _x->p[4], _x->p[5] );
				pose_rel = float_camera_extrinsic_compose( pose_rel, pose_inc );
				// Check on the convergence
				if( _md > md || _md < 1e-5 ) it = 1;
				md = _md;
				rt = true;
				nwf_vec_delete( _x );
#endif
				//else {
				//	nw_cui_error( "cu_icp_projective_3d - rcond = %f", rcond );
				//}
			}
		}
		nwf_mat_delete( AtA );
		nwf_vec_delete( Atb );
	} while( --it && rt );
	// make inlier map
	if( rt ) {
		*r_ext = pose_rel;
		if( d_inlier ) {
			int threads_per_block = CU_BLOCK_MAX;
			int blocks_per_grid = int_div_up(v1_size,threads_per_block);
			d_icp_projective_3d_step3<<<blocks_per_grid, threads_per_block>>>( v1_size, d_v21, d_inlier );
		}
	}

	cu_free( d_AtA );
	cu_free( d_Atb );
	cu_free( d_A );
	cu_free( d_b );
	cu_free( d_t );
	cu_free( d_v21 );

	return rt;
}

void cu_voxel_run_length_write_header( cu_voxel *v, FILE *fp )
{
	fwrite( VOLUME_FORMAT, sizeof(char), 16, fp );
	fwrite( &v->size, sizeof(int), 3, fp );
	fwrite( &v->min_t, sizeof(float), 1, fp );
	fwrite( &v->max_t, sizeof(float), 1, fp );
	fwrite( &v->grid_s, sizeof(float), 1, fp );
}

void cu_voxel_run_length_read_header( cu_voxel *v, FILE *fp )
{
	char format[16];
	fread( format, sizeof(char), 16, fp );
	if( strcmp( format, VOLUME_FORMAT ) == 0 ) {
		fread( &v->size, sizeof(int), 3, fp );
		fread( &v->min_t, sizeof(float), 1, fp );
		fread( &v->max_t, sizeof(float), 1, fp );
		fread( &v->grid_s, sizeof(float), 1, fp );
		v->size_xy = v->size.x * v->size.y;
	}
	else {
		nw_cui_error( "cu_voxel_run_length_decode_header - invalid format." );
	}
}

typedef struct _cu_run_length_dump {
	uint	n;
	uint	*size_p;
	char	**data_pp;
} cu_run_length_dump;

void cu_voxel_run_length_read_dump( cu_run_length_dump *p, FILE *fp )
{
	p->size_p = alloc_uint( p->n );
	p->data_pp = alloc_x( char*, p->n );
	for( int i=0; i<p->n; i++ ) {
		fread( p->size_p+i, sizeof(uint), 1, fp );
		p->data_pp[i] = alloc_char( p->size_p[i] );
		fread( p->data_pp[i], sizeof(char), p->size_p[i], fp );
	}
}

void cu_voxel_run_length_write_dump( cu_run_length_dump *p, FILE *fp )
{
	for( int i=0; i<p->n; i++ ) {
		fwrite( p->size_p+i, sizeof(uint), 1, fp );
		fwrite( p->data_pp[i], sizeof(char), p->size_p[i], fp );
	}
}

void cu_voxel_run_length_free_dump( cu_run_length_dump *p )
{
	for( int i=0; i<p->n; i++ ) {
		free( p->data_pp[i] );
	}
	free( p->size_p );
	free( p->data_pp );
}

typedef struct _CU_VOXEL_RLE {
	ushort	n : 14;
	ushort	v : 2;
} cu_voxel_rle;

void cu_voxel_run_length_encode_slice( char *buffer, uint *buffer_size, cu_voxel_t *data_s, cu_voxel_t *data_e )
{
	const uint	voxel1_size = sizeof(cu_voxel_rle);
	const uint	voxel2_size = sizeof(cu_voxel_t);
	const uint	voxel3_size = voxel1_size + voxel2_size;

	uint		buffer_count = 0, data_count = 0;
	cu_voxel_t	data_cur, data_old;
	cu_voxel_rle	rle;
	data_old = *data_s;
	do {
		data_cur = *data_s;
		if( data_cur.tsdf == data_old.tsdf ) {
			if( ++data_count > 0x3fff ) --data_count;
			else continue;
		}
		rle.n = data_count;
		if( data_old.tsdf == CU_VOXEL_MAX_TSDF ) {
			rle.v = 0x1;
			*(cu_voxel_rle*)buffer = rle; buffer += voxel1_size; buffer_count += voxel1_size;
		}
		else if( data_old.tsdf == -CU_VOXEL_MAX_TSDF ) {
			rle.v = 0x2;
			*(cu_voxel_rle*)buffer = rle; buffer += voxel1_size; buffer_count += voxel1_size;
		}
		else {
			rle.v = 0x3;
			*(cu_voxel_rle*)buffer = rle; buffer += voxel1_size;
			*(cu_voxel_t*)buffer = data_old; buffer += voxel2_size;
			buffer_count += voxel3_size;
		}
		data_count = 1;
		data_old = data_cur;
	} while( ++data_s != data_e );
	rle.n = data_count;
	if( data_old.tsdf == CU_VOXEL_MAX_TSDF ) {
		rle.v = 0x1;
		*(cu_voxel_rle*)buffer = rle; buffer += voxel1_size; buffer_count += voxel1_size;
	}
	else if( data_old.tsdf == -CU_VOXEL_MAX_TSDF ) {
		rle.v = 0x2;
		*(cu_voxel_rle*)buffer = rle; buffer += voxel1_size; buffer_count += voxel1_size;
	}
	else {
		rle.v = 0x3;
		*(cu_voxel_rle*)buffer = rle; buffer += voxel1_size;
		*(cu_voxel_t*)buffer = data_old; buffer += voxel2_size;
		buffer_count += voxel3_size;
	}
	*buffer_size = buffer_count;
}

void cu_voxel_run_length_decode_slice( char *buffer, cu_voxel_t *data_s, cu_voxel_t *data_e )
{
	const uint	voxel1_size = sizeof(cu_voxel_rle);
	const uint	voxel2_size = sizeof(cu_voxel_t);

	const cu_voxel_t	tsdf_empty = {CU_VOXEL_MAX_TSDF, 0, make_uchar3(0,0,0)};
	const cu_voxel_t	tsdf_null = {-CU_VOXEL_MAX_TSDF, 0, make_uchar3(0,0,0)};
	cu_voxel_t		data_old;
	cu_voxel_rle	rle;
	rle = *(cu_voxel_rle*)buffer; buffer += voxel1_size;
	if( rle.v == 0x1 )		data_old = tsdf_empty;
	else if( rle.v == 0x2 ) data_old = tsdf_null;
	else {
		data_old = *(cu_voxel_t*)buffer; buffer += voxel2_size;
	}
	do {
		if( rle.n == 0 ) {
			rle = *(cu_voxel_rle*)buffer; buffer += voxel1_size;
			if( rle.v == 0x1 )		data_old = tsdf_empty;
			else if( rle.v == 0x2 ) data_old = tsdf_null;
			else {
				data_old = *(cu_voxel_t*)buffer; buffer += voxel2_size;
			}
		}
		do {
			*data_s++ = data_old;
		} while( --rle.n );
	} while( data_s != data_e );
}

/// Note that v must be allocated.

void cu_voxel_run_length_encode( cu_voxel *v_local, char *file_name, int3 begin_local, int3 begin_global, int3 size )
{
	// open file
	FILE	*fp;
	fp = fopen( file_name, "rb" );
	if( fp == NULL ) {
		nw_cui_message_s( "cu_voxel_run_length_encode - saving..." );
		cu_array_host<cu_voxel_t> h_data_local( v_local->size_xy );
		char *buffer = alloc_char( v_local->size_xy * sizeof(cu_voxel_t) * sizeof(cu_voxel_rle) );
		cu_run_length_dump	d;
		d.n = v_local->size.z;
		d.size_p = alloc_uint( d.n );
		d.data_pp = alloc_x( char*, d.n );
		for( int z=0; z<d.n; z++ ) {
			cu_memcpy_device_to_host( cu_voxel_access_z(v_local,z), h_data_local.ptr(), v_local->size_xy );
			cu_voxel_run_length_encode_slice( buffer, &d.size_p[z], h_data_local.ptr(), h_data_local.ptr() + h_data_local.size() );
			d.data_pp[z] = alloc_char( d.size_p[z] );
			copy_char( d.data_pp[z], buffer, d.size_p[z] );
		}
		free( buffer );
		fp = fopen( file_name, "wb" );
		cu_voxel_run_length_write_header( v_local, fp );
		cu_voxel_run_length_write_dump( &d, fp );
		cu_voxel_run_length_free_dump( &d );
		fclose( fp );
		nw_cui_message_e( "done" );
		return;
	}
	// check boundary
	int3	end_local = begin_local + size;
	if( end_local.x > v_local->size.x || end_local.y > v_local->size.y || end_local.z > v_local->size.z ) {
		nw_cui_error( "cu_voxel_run_length_encode - out of bound." );
		fclose( fp );
		return;
	}
	// read header
	cu_voxel	v_global;
	cu_voxel_run_length_read_header( &v_global, fp );
	// check boundary
	int3	end_global = begin_global + size;
	if( end_global.x > v_global.size.x || end_global.y > v_global.size.y || end_global.z > v_global.size.z ||
		v_local->grid_s != v_global.grid_s ) {
			nw_cui_error( "cu_voxel_run_length_encode - out of bound." );
			fclose( fp );
			return;
	}
	nw_cui_message_s( "cu_voxel_run_length_encode - updating..." );
	// read dump
	cu_run_length_dump	d;
	d.n = v_global.size.z;
	cu_voxel_run_length_read_dump( &d, fp );
	fclose( fp );
	// update dump
	cu_voxel_t	*h_data_local = alloc_x( cu_voxel_t, v_local->size_xy );
	cu_voxel_t	*h_data_global = alloc_x( cu_voxel_t, v_global.size_xy );
	char		*buffer = alloc_char( v_global.size_xy * 10 );
	for( int z=0; z<size.z; z++ ) {
		checkCudaErrors(cudaMemcpy( h_data_local, v_local->data+(z+begin_local.z)*v_local->size_xy, v_local->size_xy*sizeof(cu_voxel_t), cudaMemcpyDeviceToHost ));
		cu_voxel_run_length_decode_slice( d.data_pp[z+begin_global.z], h_data_global, h_data_global + v_global.size_xy );
		for( int y=0; y<size.y; y++ ) {
			copy_x( cu_voxel_t, h_data_global+(y+begin_global.y)*v_global.size.x+begin_global.x,
				h_data_local+(y+begin_local.y)*v_local->size.x+begin_local.x, size.x );
		}
		cu_voxel_run_length_encode_slice( buffer, d.size_p+z+begin_global.z, h_data_global, h_data_global + v_global.size_xy );
		free( d.data_pp[z+begin_global.z] );
		d.data_pp[z+begin_global.z] = alloc_char( d.size_p[z+begin_global.z] );
		copy_char( d.data_pp[z+begin_global.z], buffer, d.size_p[z+begin_global.z] );
	}
	free( h_data_local );
	free( h_data_global );
	free( buffer );
	// write after update
	fp = fopen( file_name, "wb" );
	cu_voxel_run_length_write_header( &v_global, fp );
	cu_voxel_run_length_write_dump( &d, fp );
	cu_voxel_run_length_free_dump( &d );
	fclose( fp );
	nw_cui_message_e( "done" );
}

/// Note that v must be allocated.

void cu_voxel_run_length_decode( cu_voxel *v_local, char *file_name, int3 begin_local, int3 begin_global, int3 size )
{
	// open file
	FILE	*fp;
	fp = fopen( file_name, "rb" );
	if( fp == NULL ) {
		nw_cui_message_s( "cu_voxel_run_length_decode - fail to open %s file.", file_name );
		return;
	}
	if( v_local->data == NULL ) {
		nw_cui_message_s( "cu_voxel_run_length_decode - loading..." );
		// read header
		cu_voxel_run_length_read_header( v_local, fp );
		cu_malloc( &v_local->data, v_local->size_xy * v_local->size.z );
		// read dump
		cu_run_length_dump	d;
		d.n = v_local->size.z;
		cu_voxel_run_length_read_dump( &d, fp );
		fclose( fp );
		// loading
		cu_array_host<cu_voxel_t> h_data_local( v_local->size_xy );
		for( int z=0; z<v_local->size.z; z++ ) {
			cu_voxel_run_length_decode_slice( d.data_pp[z], h_data_local.ptr(), h_data_local.ptr() + h_data_local.size() );
			cu_memcpy_host_to_device( h_data_local.ptr(), cu_voxel_access_z(v_local,z), v_local->size_xy );
		}
		cu_voxel_run_length_free_dump( &d );
		nw_cui_message_e( "done" );
		nw_cui_debug("min_t = %f, max_t = %f, grid_s = %f", v_local->min_t, v_local->max_t, v_local->grid_s );
		nw_cui_debug("width = %d, height = %d, depth = %d", v_local->size.x, v_local->size.y, v_local->size.z);
		return;
	}
	// check boundary
	int3	end_local = begin_local + size;
	if( end_local.x > v_local->size.x || end_local.y > v_local->size.y || end_local.z > v_local->size.z ) {
		nw_cui_error( "cu_voxel_run_length_decode - out of bound." );
		fclose( fp );
		return;
	}
	// read header
	cu_voxel	v_global;
	cu_voxel_run_length_read_header( &v_global, fp );
	// check boundary
	int3	end_global = begin_global + size;
	if( end_global.x > v_global.size.x || end_global.y > v_global.size.y || end_global.z > v_global.size.z ||
		v_local->grid_s != v_global.grid_s ) {
			nw_cui_error( "cu_voxel_run_length_decode - out of bound." );
			fclose( fp );
			return;
	}
	nw_cui_message_s( "cu_voxel_run_length_decode - updating..." );
	// read dump
	cu_run_length_dump	d;
	d.n = v_global.size.z;
	cu_voxel_run_length_read_dump( &d, fp );
	fclose( fp );
	// load global to local
	cu_voxel_t	*h_data_global = alloc_x( cu_voxel_t, v_global.size_xy );
	for( int z=0; z<size.z; z++ ) {
		cu_voxel_run_length_decode_slice( d.data_pp[z+begin_global.z], h_data_global, h_data_global + v_global.size_xy );
		for( int y=0; y<size.y; y++ ) {
			checkCudaErrors(cudaMemcpy( cu_voxel_access_yz(v_local, y+begin_local.y, z+begin_local.z)+begin_local.x,
				h_data_global+(y+begin_global.y)*v_global.size.x+begin_global.x, size.x*sizeof(cu_voxel_t), cudaMemcpyHostToDevice ));
		}
	}
	free( h_data_global );
	cu_voxel_run_length_free_dump( &d );
	nw_cui_message_e( "done" );
}

void __global__ d_voxel_rlc_to_point_cloud_kernel( int z, int3 size, cu_voxel_t *data1_ptr, cu_voxel_t *data2_ptr,
	float3 *vertex_ptr, float3 *normal_ptr, uchar3 *color_ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( x >= size.x || y >= size.y || z >= size.z )
		return;
	int		index = ( y * size.x + x );
	vertex_ptr[index].z = FLT_MAX;
	if( x == (size.x-1) || y == (size.y-1) || z == (size.z-1) )
		return;
	float	tsdf_0 = d_unpack_tsdf( data1_ptr[index].tsdf );
	float	tsdf[7];
	tsdf[0] = d_unpack_tsdf( data1_ptr[index+1].tsdf );
	tsdf[1] = d_unpack_tsdf( data1_ptr[index+size.x+1].tsdf );
	tsdf[2] = d_unpack_tsdf( data1_ptr[index+size.x].tsdf );
	tsdf[3] = d_unpack_tsdf( data2_ptr[index+1].tsdf );
	tsdf[4] = d_unpack_tsdf( data2_ptr[index+size.x+1].tsdf );
	tsdf[5] = d_unpack_tsdf( data2_ptr[index+size.x].tsdf );
	tsdf[6] = d_unpack_tsdf( data2_ptr[index].tsdf );
	int3	voxel_0 = make_int3( x, y, z );
	int3	voxel[7];
	voxel[0] = make_int3( x+1, y, z );
	voxel[1] = make_int3( x+1, y+1, z );
	voxel[2] = make_int3( x, y+1, z );
	voxel[3] = make_int3( x+1, y, z+1 );
	voxel[4] = make_int3( x+1, y+1, z+1 );
	voxel[5] = make_int3( x, y+1, z+1 );
	voxel[6] = make_int3( x, y, z+1 );
	float	dot_min = FLT_MAX, dot, dt, t;
	float3	t_pos, n_pos, t_pos_, n_pos_, ray_d;
	uchar3	c_pos;
	int3	i_pos;
	for( int i=0; i<7; i++ ) {
		// zero crossing check
		if( tsdf[i]<0.0f && tsdf[i]>-1.0f && tsdf_0>=0.0f && tsdf_0<1.0f ) {
			vector3_sub( ray_d, voxel[i], voxel_0 );
			dt = float3_norm2( ray_d );
			ray_d = float3_div_c( ray_d, dt );
			t = dt * tsdf_0 / ( tsdf_0 - tsdf[i] );
			t_pos_ = float3_mul_c( ray_d, t );
			vector3_add( t_pos_, t_pos_, voxel_0 );
			vector3_copy( i_pos, t_pos_ );
			if( i_pos.x<0 || i_pos.x>=(size.x-1) || i_pos.y<0 || i_pos.y>=(size.y-1) || (i_pos.z != voxel_0.z) )
				continue;
			n_pos_ = _d_voxel_extract_surface_normal( size.x, data1_ptr, data2_ptr, t_pos_, i_pos );
			dot = float3_dot( n_pos_, ray_d );
			if( dot < dot_min ) {
				dot_min = dot; t_pos = t_pos_; n_pos = n_pos_;
				c_pos = _d_voxel_extract_surface_color( size.x, data1_ptr, data2_ptr, t_pos_, i_pos );
			}
		}
	}
	for( int i=0; i<7; i++ ) {
		// zero crossing check
		if( tsdf_0<0.0f && tsdf_0>-1.0f && tsdf[i]>=0.0f && tsdf[i]<1.0f ) {
			vector3_sub( ray_d, voxel_0, voxel[i] );
			dt = float3_norm2( ray_d );
			ray_d = float3_div_c( ray_d, dt );
			t = dt * tsdf[i] / ( tsdf[i] - tsdf_0 );
			t_pos_ = float3_mul_c( ray_d, t );
			vector3_add( t_pos_, t_pos_, voxel[i] );
			vector3_copy( i_pos, t_pos_ );
			if( i_pos.x<0 || i_pos.x>=(size.x-1) || i_pos.y<0 || i_pos.y>=(size.y-1) || (i_pos.z != voxel_0.z) )
				continue;
			n_pos_ = _d_voxel_extract_surface_normal( size.x, data1_ptr, data2_ptr, t_pos_, i_pos );
			dot = float3_dot( n_pos_, ray_d );
			if( dot < dot_min ) {
				dot_min = dot; t_pos = t_pos_; n_pos = n_pos_;
				c_pos = _d_voxel_extract_surface_color( size.x, data1_ptr, data2_ptr, t_pos_, i_pos );
			}
		}
	}
	if( dot_min < 0.0f ) {
		vertex_ptr[index] = t_pos;
		normal_ptr[index] = n_pos;
		color_ptr[index] = c_pos;
	}
}

void cu_voxel_rlc_to_point_cloud( char *file_input, char *file_output )
{
	// open file
	FILE	*fp;
	fp = fopen( file_input, "rb" );
	if( fp == NULL ) {
		nw_cui_message_s( "cu_voxel_rlc_to_point_cloud - fail to open %s file.", file_input );
		return;
	}
	// read header
	cu_voxel	volume;
	cu_voxel_run_length_read_header( &volume, fp );
	nw_cui_debug("min_t = %f, max_t = %f, grid_s = %f", volume.min_t, volume.max_t, volume.grid_s );
	nw_cui_debug("width = %d, height = %d, depth = %d", volume.size.x, volume.size.y, volume.size.z);
	volume.size_xy = volume.size.x * volume.size.y;
	checkCudaErrors(cudaMalloc( &volume.data, volume.size_xy * 2 * sizeof(cu_voxel_t) )); // FIXME
	// read dump
	nw_cui_message_s( "cu_voxel_rlc_to_point_cloud - loading..." );
	cu_run_length_dump	d;
	d.n = volume.size.z;
	cu_voxel_run_length_read_dump( &d, fp );
	fclose( fp );
	nw_cui_message_e( "done" );
	// converting
	nw_cui_message_s( "cu_voxel_rlc_to_point_cloud - converting..." );

	cu_voxel_t	*h_data = alloc_x( cu_voxel_t, volume.size_xy );
	cu_voxel_t	*d_data_ptr[2] = { volume.data, volume.data+volume.size_xy };

	float3	*h_vertex = alloc_x( float3, volume.size_xy );
	float3	*h_normal = alloc_x( float3, volume.size_xy );
	uchar3	*h_color = alloc_x( uchar3, volume.size_xy );

	float3	*d_vertex, *d_normal;
	checkCudaErrors(cudaMalloc( &d_vertex, volume.size_xy * sizeof(float3) ));
	checkCudaErrors(cudaMalloc( &d_normal, volume.size_xy * sizeof(float3) ));
	uchar3	*d_color;
	checkCudaErrors(cudaMalloc( &d_color, volume.size_xy * sizeof(uchar3) ));

	dim3	block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	dim3	grid = dim3(int_div_up(volume.size.x,block.x), int_div_up(volume.size.y,block.y));

	nw_stack	*stack = nw_stack_new( 100000, sizeof(float3)*2+sizeof(uchar3) );
	for( int z=0; z<volume.size.z; z++ ) {
		cu_voxel_run_length_decode_slice( d.data_pp[z], h_data, h_data + volume.size_xy );
		if( z==0 ) {
			checkCudaErrors(cudaMemcpy( d_data_ptr[0], h_data, volume.size_xy*sizeof(cu_voxel_t), cudaMemcpyHostToDevice ));
			continue;
		}
		checkCudaErrors(cudaMemcpy( d_data_ptr[1], h_data, volume.size_xy*sizeof(cu_voxel_t), cudaMemcpyHostToDevice ));
		d_voxel_rlc_to_point_cloud_kernel<<<grid,block>>>( z-1, volume.size, d_data_ptr[0], d_data_ptr[1], d_vertex, d_normal, d_color );
		cu_error(); cu_sync();
		checkCudaErrors(cudaMemcpy( h_vertex, d_vertex, volume.size_xy * sizeof(float3), cudaMemcpyDeviceToHost ));
		checkCudaErrors(cudaMemcpy( h_normal, d_normal, volume.size_xy * sizeof(float3), cudaMemcpyDeviceToHost ));
		checkCudaErrors(cudaMemcpy( h_color, d_color, volume.size_xy * sizeof(uchar3), cudaMemcpyDeviceToHost ));
		for( int i=0; i<volume.size_xy; i++ ) {
			if( h_vertex[i].z == FLT_MAX )
				continue;
			float3	*ptr_ = (float3*)_nw_stack_push( stack );
			ptr_[0] = float3_mul_c( h_vertex[i], volume.grid_s );
			ptr_[1] = h_normal[i];
			*(uchar3*)(ptr_+2) = h_color[i];
		}
		vector1_swap( d_data_ptr[0], d_data_ptr[1] );
	}
	checkCudaErrors(cudaFree( volume.data ));
	checkCudaErrors(cudaFree( d_vertex ));
	checkCudaErrors(cudaFree( d_normal ));
	checkCudaErrors(cudaFree( d_color ));
	free( h_vertex );
	free( h_normal );
	free( h_color );
	free( h_data );
	cu_voxel_run_length_free_dump( &d );
	nw_cui_message_e( "done" );

  char *ext_ = strrchr(file_output,'.');
	if( strcmp( ext_, ".ply" ) == 0 ) {
		// write to ply file
		nw_cui_message_s( "cu_voxel_rlc_to_point_cloud - write to ply..." );
		fp = fopen( file_output, "wb" );
		fprintf( fp, "ply\n" );
		fprintf( fp, "format binary_little_endian 1.0\n" );
		fprintf( fp, "element vertex %d\n", stack->n );
		fprintf( fp, "property float32 x\n" );
		fprintf( fp, "property float32 y\n" );
		fprintf( fp, "property float32 z\n" );
		fprintf( fp, "property float32 nx\n" );
		fprintf( fp, "property float32 ny\n" );
		fprintf( fp, "property float32 nz\n" );
		fprintf( fp, "property uchar diffuse_blue\n" );
		fprintf( fp, "property uchar diffuse_green\n" );
		fprintf( fp, "property uchar diffuse_red\n" );
		fprintf( fp, "end_header\n" );
		void *ptr_;
		while( ptr_ = nw_stack_pop(stack) ) {
			fwrite( ptr_, 27, 1, fp );
		}
		fclose( fp );
		nw_cui_message_e( "done" );
	}
	else if(strcmp( ext_, ".obj" ) == 0 ) {
		// write to obj file
		nw_cui_message_s( "cu_voxel_rlc_to_point_cloud - write to obj..." );
		fp = fopen( file_output, "w" );
		float3 *ptr_;
		while( ptr_ = (float3*)nw_stack_pop(stack) ) {
			fprintf( fp, "vn %f %f %f\n", ptr_[1].x, ptr_[1].y, ptr_[1].z );
			//fprintf( fp, "vc %d %d %d\n", ((uchar3*)(ptr_+2))->x, ((uchar3*)(ptr_+2))->y, ((uchar3*)(ptr_+2))->z );
			fprintf( fp, "v %f %f %f\n", ptr_[0].x, ptr_[0].y, ptr_[0].z );
		}
		fclose( fp );
		nw_cui_message_e( "done" );
	}
	else {
		nw_cui_error( "cu_voxel_rlc_to_point_cloud - Invalid file extension (%s)", ext_ );
	}
	nw_stack_delete( stack );
}

__global__ void d_voxel_extract_surface_normals( cu_voxel v, int size, float3 *vertex_ptr, float3 *normal_ptr )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i >= size )
		return;
	normal_ptr[i] = d_voxel_extract_surface_normal( &v, float3_div_c( vertex_ptr[i], v.grid_s ) ); // FIXME
}

#include "cu_marching_cubes_v1.h"
void cu_voxel_rlc_to_mesh( char *file_input, char *file_output )
{
	cu_voxel	volume; volume.data = NULL;
	cu_voxel_run_length_decode( &volume, file_input, make_int3(0,0,0), make_int3(0,0,0), make_int3(0,0,0) );

	nw_cui_message_s( "cu_voxel_rlc_to_mesh - generating triangles..." );
	MarchingCubes mc;
	cu_array<float3> triangles_buffer; // If a memory error returns, adjust DEFAULT_TRIANGLES_BUFFER_SIZE!
	cu_array<float3> triangles_device = mc.run( volume, triangles_buffer );
	nw_cui_message_e( "done" );
	nw_cui_message( "cu_voxel_rlc_to_mesh - # of triangles = %d", triangles_device.size()/3 );

	nw_cui_message_s( "cu_voxel_rlc_to_mesh - generating normals..." );
	cu_array<float3>	normals_buffer(triangles_device.size());
	dim3	block = dim3( CU_BLOCK_MAX );
	dim3	grid = dim3( int_div_up(normals_buffer.size(), block.x) );
	d_voxel_extract_surface_normals<<<grid,block>>>( volume, normals_buffer.size(), triangles_device.ptr(), normals_buffer.ptr() );
	nw_cui_message_e( "done" );

	cu_voxel_delete( &volume );

	float3	*h_vertices = alloc_x( float3, triangles_device.size() );
	triangles_device.download( h_vertices );

	float3	*h_normals = alloc_x( float3, normals_buffer.size() );
	normals_buffer.download( h_normals );

  char *ext_ = strrchr(file_output,'.');
	if( strcmp( ext_, ".ply" ) == 0 ) {
		// write to ply file
		nw_cui_message_s( "cu_voxel_rlc_to_mesh - write to a ply file..." );
		FILE *fp = fopen( file_output, "wb" );
		fprintf( fp, "ply\n" );
		fprintf( fp, "format binary_little_endian 1.0\n" );
		fprintf( fp, "element vertex %d\n", triangles_device.size() );
		fprintf( fp, "property float32 x\n" );
		fprintf( fp, "property float32 y\n" );
		fprintf( fp, "property float32 z\n" );
		fprintf( fp, "property float32 nx\n" );
		fprintf( fp, "property float32 ny\n" );
		fprintf( fp, "property float32 nz\n" );
		fprintf( fp, "element face %d\n", triangles_device.size()/3 );
		fprintf( fp, "property list uint8 int32 vertex_indices\n" );
		fprintf( fp, "end_header\n" );
		for( int i=0; i<triangles_device.size(); i++ ) {
			fwrite( h_vertices+i, sizeof(float3), 1, fp );
			fwrite( h_normals+i, sizeof(float3), 1, fp );
		}
		uchar	prop_ = 3;
		int		face_[3];
		for( int i=0; i<triangles_device.size()/3; i++ ) {
			face_[0] = i*3; face_[1] = face_[0]+1; face_[2] = face_[1]+1;
			fwrite( &prop_, 1, 1, fp );
			fwrite( face_, sizeof(int), 3, fp );
		}
		fclose( fp );
		nw_cui_message_e( "done" );
	}
	else if( strcmp( ext_, ".obj" ) == 0 ) {
		// write to obj file
		nw_cui_message_s( "cu_voxel_rlc_to_mesh - write to an obj file..." );
		FILE *fp = fopen( file_output, "w" );
		for( int i=0; i<triangles_device.size(); i++ ) {
			fprintf( fp, "v %f %f %f\n", h_vertices[i].x, h_vertices[i].y, h_vertices[i].z );
			fprintf( fp, "vn %f %f %f\n", h_normals[i].x, h_normals[i].y, h_normals[i].z );
		}
		for( int i=0; i<triangles_device.size()/3; i++ ) {
			fprintf( fp, "f %d %d %d\n", i*3+1, i*3+2, i*3+3 );
		}
		fclose( fp );
		nw_cui_message_e( "done" );
	}
	else {
		nw_cui_error( "cu_voxel_rlc_to_mesh - Invalid file extension (%s)", ext_ );
	}
	if ( h_vertices != NULL)
	{
		free( h_vertices );
	}
	if ( h_normals != NULL)
	{
		free( h_normals );
	}
	
}

void cu_kinect_fusion_new( cu_kinect_fusion_params *kf )
{
	for( int i=1; i<KINECT_DEPTH_LEVEL; i++ ) {
		kf->video_size[i] = make_int2( (kf->video_size[0].x)>>i, (kf->video_size[0].y)>>i );
		kf->depth_size[i] = make_int2( (kf->depth_size[0].x)>>i, (kf->depth_size[0].y)>>i );
		kf->video_int[i] = float_camera_intrinsic_pyramid_down( kf->video_int[i-1] );
		kf->depth_int[i] = float_camera_intrinsic_pyramid_down( kf->depth_int[i-1] );
	}
	for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
		kf->video_size_t[i] = kf->video_size[i].x * kf->video_size[i].y;
		kf->depth_size_t[i] = kf->depth_size[i].x * kf->depth_size[i].y;
	}

	kf->d_video_dmap = NULL;
	kf->d_video = NULL;
	if( kf->h_video_dmap ) {
		checkCudaErrors(cudaMalloc( &kf->d_video_dmap, kf->video_size_t[0] * sizeof(float2) ));
		checkCudaErrors(cudaMemcpy( kf->d_video_dmap, kf->h_video_dmap, kf->video_size_t[0] * sizeof(float2), cudaMemcpyHostToDevice ));
		checkCudaErrors(cudaMalloc( &kf->d_video, kf->video_size_t[0] * sizeof(uchar3) ));
	}
	kf->d_depth_dmap = NULL;
	kf->d_depth = NULL;
	kf->d_disparity = NULL;
	if( kf->h_depth_dmap ) {
		checkCudaErrors(cudaMalloc( &kf->d_depth_dmap, kf->depth_size_t[0] * sizeof(float2) ));
		checkCudaErrors(cudaMemcpy( kf->d_depth_dmap, kf->h_depth_dmap, kf->depth_size_t[0] * sizeof(float2), cudaMemcpyHostToDevice ));
		if( kf->base_line )
			cu_malloc( &kf->d_disparity, kf->depth_size_t[0] );
		else
			checkCudaErrors(cudaMalloc( &kf->d_depth, kf->depth_size_t[0] * sizeof(short) ));
	}
	kf->d_depth_to_meter = NULL;
	if( kf->h_depth_to_meter ) {
		checkCudaErrors(cudaMalloc( &kf->d_depth_to_meter, 2047 * sizeof(float) ));
		checkCudaErrors(cudaMemcpy( kf->d_depth_to_meter, kf->h_depth_to_meter, 2047 * sizeof(float), cudaMemcpyHostToDevice ));
	}
	// alloc local
	checkCudaErrors(cudaMalloc( &kf->d_video_u, kf->video_size_t[0] * sizeof(uchar3) ));
	if( kf->base_line ) {
		kf->d_depth_u = NULL;
		cu_malloc( &kf->d_disparity_u, kf->depth_size_t[0] );
	}
	else {
		kf->d_disparity_u = NULL;
		checkCudaErrors(cudaMalloc( &kf->d_depth_u, kf->depth_size_t[0] * sizeof(short) ));
	}
	checkCudaErrors(cudaMalloc( &kf->d_mask[0], kf->depth_size_t[0] * sizeof(uchar) ));
	checkCudaErrors(cudaMalloc( &kf->d_mask[1], kf->depth_size_t[0] * sizeof(uchar) ));
	checkCudaErrors(cudaMalloc( &kf->d_inlier, kf->depth_size_t[0] * sizeof(uchar) ));
	checkCudaErrors(cudaMalloc( &kf->d_meter_x, kf->depth_size_t[0] * sizeof(float) ));
	cu_malloc( &kf->d_vertex_x, kf->depth_size_t[0] );

	for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
		checkCudaErrors(cudaMalloc( &kf->d_meter_u[i], kf->depth_size_t[i] * sizeof(float) ));
		checkCudaErrors(cudaMalloc( &kf->d_vertex_u[i], kf->depth_size_t[i] * sizeof(float3) ));
		checkCudaErrors(cudaMalloc( &kf->d_normal_u[i], kf->depth_size_t[i] * sizeof(float3) ));

		checkCudaErrors(cudaMalloc( &kf->d_meter_s[i], kf->depth_size_t[i] * sizeof(float) ));
		checkCudaErrors(cudaMalloc( &kf->d_vertex_s[i], kf->depth_size_t[i] * sizeof(float3) ));
		checkCudaErrors(cudaMalloc( &kf->d_normal_s[i], kf->depth_size_t[i] * sizeof(float3) ));
		checkCudaErrors(cudaMalloc( &kf->d_color_s[i], kf->depth_size_t[i] * sizeof(uchar3) ));
	}

	//kf->icp_params.dt = 0.05f; // 50 mm
	//kf->icp_params.nt = (float)cos( nw_degree_to_radian(20) );
	//kf->icp_params.rt = nw_degree_to_radian(10); // 10 degree
	//kf->icp_params.tt = 0.05f; // 50 mm
	//nw_fill_3( kf->icp_it_max, 10, 5, 4 );

  kf->icp_params.dt = 0.10f; // 100 mm
	kf->icp_params.nt = (float)cos( nw_degree_to_radian(20) );
	kf->icp_params.rt = nw_degree_to_radian(20); // 10 degree
	kf->icp_params.tt = 0.10f; // 100 mm
	nw_fill_3( kf->icp_it_max, 15, 11, 8 );


	kf->video_depth_ext = float_camera_extrinsic_inverse( kf->depth_video_ext );

	cu_malloc_host( &kf->h_video, kf->video_size_t[0] );
	if( kf->base_line ) {
		kf->h_depth = NULL;
		cu_malloc_host( &kf->h_disparity, kf->depth_size_t[0] );
	}
	else {
		kf->h_disparity = NULL;
		cu_malloc_host( &kf->h_depth, kf->depth_size_t[0] );
	}
#ifdef CG_KLT_TRACKER
	kf->klt.create( kf->video_size[0].x, kf->video_size[0].y, true );
#endif
	// execution time measurements
	kf->time_count = 0;
	kf->time_icp = 0; kf->time_count_icp = 0;
	kf->time_of = 0; kf->time_count_of = 0;
	kf->time_surf = 0; kf->time_count_surf = 0;
	kf->time_integration = 0; kf->time_count_integration = 0;
	kf->time_extraction = 0; kf->time_count_extraction = 0;

	// debug
	//cu_malloc_host( &kf->h_image1, kf->depth_size_t[0] * 12 );
	//cu_malloc_host( &kf->h_image2, kf->depth_size_t[0] * 12 );
	//cu_malloc_host( &kf->h_image3, kf->depth_size_t[0] );
}

void cu_kinect_fusion_delete( cu_kinect_fusion_params *kf )
{
	if( kf->d_meter_u[0] ) {
		if( kf->d_video_dmap ) checkCudaErrors(cudaFree( kf->d_video_dmap ));
		if( kf->d_depth_dmap ) checkCudaErrors(cudaFree( kf->d_depth_dmap ));
		if( kf->d_depth_to_meter ) checkCudaErrors(cudaFree( kf->d_depth_to_meter ));

		// free local
		checkCudaErrors(cudaFree( kf->d_mask[0] ));
		checkCudaErrors(cudaFree( kf->d_mask[1] ));
		if( kf->d_video ) checkCudaErrors(cudaFree( kf->d_video ));
		if( kf->d_depth ) checkCudaErrors(cudaFree( kf->d_depth ));
		cu_free( kf->d_disparity );
		checkCudaErrors(cudaFree( kf->d_video_u ));
		checkCudaErrors(cudaFree( kf->d_depth_u ));
		cu_free( kf->d_disparity_u );
		checkCudaErrors(cudaFree( kf->d_inlier ));
		checkCudaErrors(cudaFree( kf->d_meter_x ));
		cu_free( kf->d_vertex_x );

		for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
			checkCudaErrors(cudaFree( kf->d_meter_u[i] ));
			checkCudaErrors(cudaFree( kf->d_vertex_u[i] ));
			checkCudaErrors(cudaFree( kf->d_normal_u[i] ));

			checkCudaErrors(cudaFree( kf->d_meter_s[i] ));
			checkCudaErrors(cudaFree( kf->d_vertex_s[i] ));
			checkCudaErrors(cudaFree( kf->d_normal_s[i] ));
			checkCudaErrors(cudaFree( kf->d_color_s[i] ));
		}

		cu_free_host( kf->h_video );
		cu_free_host( kf->h_depth );
		cu_free_host( kf->h_disparity );
		// debug
		//cu_free_host( kf->h_image1 );
		//cu_free_host( kf->h_image2 );
		//cu_free_host( kf->h_image3 );
		kf->d_meter_u[0] = NULL;
#ifdef CG_KLT_TRACKER
		kf->klt.release();
#endif
	}
}
// Given:
// kf->h_video and (kf->h_depth or kf->h_disparity)
// Possible options are follows:
// first frame:	true, false, true
// tracking:	true, true, true
// rendering:	false, false, true
// is_dynamic:	dynamic target or not
void cu_kinect_fusion_process( cu_kinect_fusion_params *kf, bool integration, bool tracking, bool extraction, bool is_dynamic )
{
	dim3	block, grid;
	bool	rt = true; // for tracking

	if( integration || tracking ) {
		if( kf->d_video_dmap ) {
			cu_memcpy_host_to_device( kf->h_video, kf->d_video, kf->video_size_t[0] );
			// undistort
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->video_size[0].x,block.x), int_div_up(kf->video_size[0].y,block.y));
			d_uchar3_mapping<<<grid,block>>>( kf->video_size[0], kf->video_size[0], kf->d_video_dmap, kf->d_video, kf->d_video_u );
			cu_error();
		}
		else {
			cu_memcpy_host_to_device( kf->h_video, kf->d_video_u, kf->video_size_t[0] );
		}
		if( kf->d_depth_dmap ) {
			if( kf->h_disparity ) {
				cu_memcpy_host_to_device( kf->h_disparity, kf->d_disparity, kf->depth_size_t[0] );
				// undistort
				block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
				grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
				d_float_mapping<<<grid,block>>>( kf->depth_size[0], kf->depth_size[0], kf->d_depth_dmap, kf->d_disparity, kf->d_disparity_u );
				cu_error();
			}
			if( kf->h_depth ) {
				cu_memcpy_host_to_device( kf->h_depth, kf->d_depth, kf->depth_size_t[0] );
				// undistort
				block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
				grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
				d_short_mapping<<<grid,block>>>( kf->depth_size[0], kf->depth_size[0], kf->d_depth_dmap, kf->d_depth, kf->d_depth_u );
				cu_error();
			}
		}
		else {
			if( kf->h_disparity )
				cu_memcpy_host_to_device( kf->h_disparity, kf->d_disparity_u, kf->depth_size_t[0] );
			if( kf->h_depth )
				cu_memcpy_host_to_device( kf->h_depth, kf->d_depth_u, kf->depth_size_t[0] );
		}
		cu_sync();
		// undistorted meter, vertex, normal (multi-scale)
		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[0], block.x) );
		// meter_u[0]
		if( kf->d_disparity_u )
			d_float_disparity_to_meter<<<grid,block>>>( kf->depth_size_t[0], kf->d_disparity_u, kf->d_meter_u[0],
			kf->base_line * kf->depth_int[0].fc.x, kf->disparity_offset );
		else
			d_kinect_depth_to_meter<<<grid,block>>>( kf->depth_size_t[0], kf->d_depth_to_meter, kf->d_depth_u, kf->d_meter_u[0] );
		cu_error(); cu_sync();
	}
	if( tracking ) {
		// depth denoising
		cu_memcpy_device_to_device( kf->d_meter_u[0], kf->d_meter_x, kf->depth_size_t[0] );
		_cu_bilateral_filtering( kf->d_meter_u[0], kf->depth_size[0].x, kf->depth_size[0].y, 3.5f, 0.03f ); // FIXME
		//_cu_nlm_filtering( kf->d_meter_u[0], kf->depth_size[0].x, kf->depth_size[0].y, 5.0f, 0.1f );

		d_float_data_to_mask<<<grid,block>>>( kf->depth_size_t[0], kf->d_meter_u[0], kf->d_mask[0] );
		cu_error(); cu_sync();
		// meter_u[1]
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[1].x,block.x), int_div_up(kf->depth_size[1].y,block.y));
		d_pyramid_down<<< grid, block>>>( kf->depth_size[0], kf->depth_size[1],
			kf->d_meter_u[0], kf->d_mask[0], kf->d_meter_u[1], kf->d_mask[1], 0.06f );
		cu_error(); cu_sync();

		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[1], block.x) );
		d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[1], kf->d_mask[1], kf->d_meter_u[1] );
		cu_error(); cu_sync();
		// meter_u[2]
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[2].x,block.x), int_div_up(kf->depth_size[2].y,block.y));
		d_pyramid_down<<< grid, block>>>( kf->depth_size[1], kf->depth_size[2],
			kf->d_meter_u[1], kf->d_mask[1], kf->d_meter_u[2], kf->d_mask[0], 0.06f );
		cu_error(); cu_sync();

		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[2], block.x) );
		d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[2], kf->d_mask[0], kf->d_meter_u[2] );
		cu_error(); cu_sync();

		for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[i].x,block.x), int_div_up(kf->depth_size[i].y,block.y));
			// vertex_u[i]
			d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[i], kf->depth_size[i], kf->d_meter_u[i], kf->d_vertex_u[i] );
			cu_error(); cu_sync();
			// normal_u[i]
			d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[i], kf->d_vertex_u[i], kf->d_normal_u[i] );
			cu_error(); cu_sync();
		}

		// debug
		//cu_memcpy_device_to_host( kf->d_normal_u[0], (float3*)kf->h_image1, kf->depth_size_t[0] );

		// We observed that the ICP tracking accuracy increased when raw depth data is used.
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
		d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[0], kf->depth_size[0], kf->d_meter_x, kf->d_vertex_u[0] );
		cu_error(); cu_sync();

		// synthesized meter, vertex, normal (multi-scale)
		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[0], block.x) );
		d_float_data_to_mask<<<grid,block>>>( kf->depth_size_t[0], kf->d_meter_s[0], kf->d_mask[0] );
		cu_error(); cu_sync();

		// meter_s[1]
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[1].x,block.x), int_div_up(kf->depth_size[1].y,block.y));
		d_pyramid_down<<< grid, block>>>( kf->depth_size[0], kf->depth_size[1],
			kf->d_meter_s[0], kf->d_mask[0], kf->d_meter_s[1], kf->d_mask[1], 0.06f );
		cu_error(); cu_sync();

		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[1], block.x) );
		d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[1], kf->d_mask[1], kf->d_meter_s[1] );
		cu_error(); cu_sync();
		// meter_s[2]
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[2].x,block.x), int_div_up(kf->depth_size[2].y,block.y));
		d_pyramid_down<<< grid, block>>>( kf->depth_size[1], kf->depth_size[2],
			kf->d_meter_s[1], kf->d_mask[1], kf->d_meter_s[2], kf->d_mask[0], 0.06f );
		cu_error(); cu_sync();

		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[2], block.x) );
		d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[2], kf->d_mask[0], kf->d_meter_s[2] );
		cu_error(); cu_sync();

		for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[i].x,block.x), int_div_up(kf->depth_size[i].y,block.y));
			// vertex_s[i]
			d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[i], kf->depth_size[i], kf->d_meter_s[i], kf->d_vertex_s[i] );
			cu_error(); cu_sync();
			// normal_s[i]
			if( i != 0 ) {
				d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[i], kf->d_vertex_s[i], kf->d_normal_s[i] );
				cu_error(); cu_sync();
			}
		}
		// Frame-model ICP tracking
		float_camera_extrinsic	relative_ext = float_camera_extrinsic_default();
		
		// Projection of the current points to the synthesized previous points is good?
		// Preliminary experiments show that the results are almost same
#if 0 
		for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
			kf->icp_params.it_max = kf->icp_it_max[i];
			rt = _cu_icp_projective_3d( kf->icp_params, kf->depth_int[i], &relative_ext, kf->depth_size[i], kf->depth_size[i],
				kf->d_vertex_u[i], kf->d_normal_u[i], kf->d_vertex_s[i], kf->d_normal_s[i], NULL );
			if( rt == false ) break;
		}
		if( rt ) {
			kf->global_ext = float_camera_extrinsic_compose( float_camera_extrinsic_inverse(relative_ext), kf->global_ext );
			//nw_cui_debug( "t = [%f %f %f]^t", kf->global_ext.t.x, kf->global_ext.t.y, kf->global_ext.t.z);
		}
#else
		for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
			kf->icp_params.it_max = kf->icp_it_max[i];
			rt = _cu_icp_projective_3d( kf->icp_params, kf->depth_int[i], &relative_ext, kf->depth_size[i], kf->depth_size[i],
				kf->d_vertex_s[i], kf->d_normal_s[i], kf->d_vertex_u[i], kf->d_normal_u[i], (i)?NULL:kf->d_inlier );
			if( rt == false ) break;
		}
		if( rt ) {
			kf->global_ext = float_camera_extrinsic_compose( relative_ext, kf->global_ext );
			nwf_SO3_coerce( (float*)kf->global_ext.R.p ); // Normalization of Rotation Matrix?
			// Skip the integration process when the current camera motion is very small (optional)
			float	w[3];	nwf_so3_from_SO3( w, (const float*)relative_ext.R.p );
			
	  if( float3_norm2( relative_ext.t ) < 0.001f && fabsf(w[0]) < 0.017f && fabsf(w[1]) < 0.017f && fabsf(w[2]) < 0.017f )
				integration = false;
			// debug
			//cu_memcpy_device_to_host( kf->d_inlier, (uchar*)kf->h_image3, kf->depth_size_t[0] );
		}
#endif
	}
	if( integration && rt ) {
		// Note that severe drifting errors are occured when the filtered depth map is used
		_cu_voxel_volumetric_integration( kf->volume, kf->depth_int[0], kf->global_ext,
			kf->depth_size[0], (tracking)?kf->d_meter_x:kf->d_meter_u[0], (tracking)?kf->d_normal_u[0]:NULL, kf->d_video_u, is_dynamic );
	}
	if( extraction ) {
		// meter_s[0], normal_s[0], color_s[0]
		_cu_voxel_extract_depth_and_normal_map( kf->volume, kf->depth_int[0], kf->global_ext,
			kf->depth_size[0], kf->d_meter_s[0], kf->d_normal_s[0], kf->d_color_s[0] );
		// debug
		//cu_memcpy_device_to_host( kf->d_normal_s[0], (float3*)kf->h_image2, kf->depth_size_t[0] );
	}
}

void cu_kinect_fusion_input( cu_kinect_fusion_params *kf )
{
	dim3	block, grid;

	if( kf->d_video_dmap ) {
		cu_memcpy_host_to_device( kf->h_video, kf->d_video, kf->video_size_t[0] );
		// undistort
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->video_size[0].x,block.x), int_div_up(kf->video_size[0].y,block.y));
		d_uchar3_mapping<<<grid,block>>>( kf->video_size[0], kf->video_size[0], kf->d_video_dmap, kf->d_video, kf->d_video_u );
		cu_error();
	}
	else {
		cu_memcpy_host_to_device( kf->h_video, kf->d_video_u, kf->video_size_t[0] );
	}
	if( kf->d_depth_dmap ) {
		if( kf->h_disparity ) {
			cu_memcpy_host_to_device( kf->h_disparity, kf->d_disparity, kf->depth_size_t[0] );
			// undistort
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
			d_float_mapping<<<grid,block>>>( kf->depth_size[0], kf->depth_size[0], kf->d_depth_dmap, kf->d_disparity, kf->d_disparity_u );
			cu_error();
		}
		if( kf->h_depth ) {
			cu_memcpy_host_to_device( kf->h_depth, kf->d_depth, kf->depth_size_t[0] );
			// undistort
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
			d_short_mapping<<<grid,block>>>( kf->depth_size[0], kf->depth_size[0], kf->d_depth_dmap, kf->d_depth, kf->d_depth_u );
			cu_error();
		}
	}
	else {
		if( kf->h_disparity )
			cu_memcpy_host_to_device( kf->h_disparity, kf->d_disparity_u, kf->depth_size_t[0] );
		if( kf->h_depth )
			cu_memcpy_host_to_device( kf->h_depth, kf->d_depth_u, kf->depth_size_t[0] );
	}
	cu_sync();
	// undistorted meter, vertex, normal (multi-scale)
	block = dim3( CU_BLOCK_MAX );
	grid = dim3( int_div_up(kf->depth_size_t[0], block.x) );
	// meter_u[0]
	if( kf->d_disparity_u )
		d_float_disparity_to_meter<<<grid,block>>>( kf->depth_size_t[0], kf->d_disparity_u, kf->d_meter_u[0],
		kf->base_line * kf->depth_int[0].fc.x, kf->disparity_offset );
	else
		d_kinect_depth_to_meter<<<grid,block>>>( kf->depth_size_t[0], kf->d_depth_to_meter, kf->d_depth_u, kf->d_meter_u[0] );
	cu_error(); cu_sync();
	// depth denoising
	cu_memcpy_device_to_device( kf->d_meter_u[0], kf->d_meter_x, kf->depth_size_t[0] );
	_cu_bilateral_filtering( kf->d_meter_u[0], kf->depth_size[0].x, kf->depth_size[0].y, 3.5f, 0.03f ); // FIXME
	//_cu_nlm_filtering( kf->d_meter_u[0], kf->depth_size[0].x, kf->depth_size[0].y, 5.0f, 0.1f ); 

	d_float_data_to_mask<<<grid,block>>>( kf->depth_size_t[0], kf->d_meter_u[0], kf->d_mask[0] );
	cu_error(); cu_sync();
	// meter_u[1]
	block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	grid = dim3(int_div_up(kf->depth_size[1].x,block.x), int_div_up(kf->depth_size[1].y,block.y));
	d_pyramid_down<<< grid, block>>>( kf->depth_size[0], kf->depth_size[1],
		kf->d_meter_u[0], kf->d_mask[0], kf->d_meter_u[1], kf->d_mask[1], 0.06f );
	cu_error(); cu_sync();

	block = dim3( CU_BLOCK_MAX );
	grid = dim3( int_div_up(kf->depth_size_t[1], block.x) );
	d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[1], kf->d_mask[1], kf->d_meter_u[1] );
	cu_error(); cu_sync();
	// meter_u[2]
	block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	grid = dim3(int_div_up(kf->depth_size[2].x,block.x), int_div_up(kf->depth_size[2].y,block.y));
	d_pyramid_down<<< grid, block>>>( kf->depth_size[1], kf->depth_size[2],
		kf->d_meter_u[1], kf->d_mask[1], kf->d_meter_u[2], kf->d_mask[0], 0.06f );
	cu_error(); cu_sync();

	block = dim3( CU_BLOCK_MAX );
	grid = dim3( int_div_up(kf->depth_size_t[2], block.x) );
	d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[2], kf->d_mask[0], kf->d_meter_u[2] );
	cu_error(); cu_sync();

	for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[i].x,block.x), int_div_up(kf->depth_size[i].y,block.y));
		// vertex_u[i]
		d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[i], kf->depth_size[i], kf->d_meter_u[i], kf->d_vertex_u[i] );
		cu_error(); cu_sync();
		// normal_u[i]
		d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[i], kf->d_vertex_u[i], kf->d_normal_u[i] );
		cu_error(); cu_sync();
	}
	//block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	//grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
	//d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[0], kf->depth_size[0], kf->d_meter_x, kf->d_vertex_u[0] );
	cu_error(); cu_sync();
}

void cu_kinect_fusion_render( cu_kinect_fusion_params *kf )
{
	dim3	block, grid;

	// meter_s[0], normal_s[0], color_s[0]
	_cu_voxel_extract_depth_and_normal_map( kf->volume, kf->depth_int[0], kf->global_ext,
		kf->depth_size[0], kf->d_meter_s[0], kf->d_normal_s[0], kf->d_color_s[0] );

	// debug
	//cu_memcpy_device_to_host( kf->d_normal_s[0], (float3*)kf->h_image2, kf->depth_size_t[0] );

	// synthesized meter, vertex, normal (multi-scale)
	block = dim3( CU_BLOCK_MAX );
	grid = dim3( int_div_up(kf->depth_size_t[0], block.x) );
	d_float_data_to_mask<<<grid,block>>>( kf->depth_size_t[0], kf->d_meter_s[0], kf->d_mask[0] );
	cu_error(); cu_sync();

	// meter_s[1]
	block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	grid = dim3(int_div_up(kf->depth_size[1].x,block.x), int_div_up(kf->depth_size[1].y,block.y));
	d_pyramid_down<<< grid, block>>>( kf->depth_size[0], kf->depth_size[1],
		kf->d_meter_s[0], kf->d_mask[0], kf->d_meter_s[1], kf->d_mask[1], 0.06f );
	cu_error(); cu_sync();

	block = dim3( CU_BLOCK_MAX );
	grid = dim3( int_div_up(kf->depth_size_t[1], block.x) );
	d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[1], kf->d_mask[1], kf->d_meter_s[1] );
	cu_error(); cu_sync();
	// meter_s[2]
	block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	grid = dim3(int_div_up(kf->depth_size[2].x,block.x), int_div_up(kf->depth_size[2].y,block.y));
	d_pyramid_down<<< grid, block>>>( kf->depth_size[1], kf->depth_size[2],
		kf->d_meter_s[1], kf->d_mask[1], kf->d_meter_s[2], kf->d_mask[0], 0.06f );
	cu_error(); cu_sync();

	block = dim3( CU_BLOCK_MAX );
	grid = dim3( int_div_up(kf->depth_size_t[2], block.x) );
	d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[2], kf->d_mask[0], kf->d_meter_s[2] );
	cu_error(); cu_sync();

	for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[i].x,block.x), int_div_up(kf->depth_size[i].y,block.y));
		// vertex_s[i]
		d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[i], kf->depth_size[i], kf->d_meter_s[i], kf->d_vertex_s[i] );
		cu_error(); cu_sync();
		// normal_s[i]
		if( i != 0 ) {
			d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[i], kf->d_vertex_s[i], kf->d_normal_s[i] );
			cu_error(); cu_sync();
		}
	}
}

/////////////////
// Experiments //
/////////////////

__global__ void d_kinect_depth_to_meter( const ptr_step_size<short> depth_ptr, ptr_step<float> meter_ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ( x >= depth_ptr.cols || y >= depth_ptr.rows )
		return;
#if defined(DEVICE_OPENNI)
	meter_ptr.ptr(y)[x] = (float)depth_ptr.ptr(y)[x] / 1000.0f;
#elif defined(DEVICE_KINECT)
	meter_ptr.ptr(y)[x] = (float)(depth_ptr.ptr(y)[x]>>3) / 1000.0f;
#endif
	// Limit valid range from 0.5 to 5 meter
	if( meter_ptr.ptr(y)[x] < KINECT_DEPTH_MIN || meter_ptr.ptr(y)[x] > KINECT_DEPTH_MAX ) meter_ptr.ptr(y)[x] = 0.0f;
}

__global__ void d_float_meter_to_gradient( const ptr_step_size<float> meter_ptr, ptr_step<float2> gradient_ptr, int r, float sigma )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ( x < r || x >= (meter_ptr.cols-r) || y < r || y >= (meter_ptr.rows-r) )
		return;
	float2	gradient = float2_set( 0 ); // FIXME
	float22	AtA = float22_set( 0 );
	float2	Atb = float2_set( 0 );
	for( int i=y-r; i<=(y+r); i++ ) {
		for( int j=x-r; j<=(x+r); j++ ) {
			float	dd = meter_ptr.ptr(i)[j] - meter_ptr.ptr(y)[x];
			// we ignore the contributions of pixels whose depth difference with the central pixel is above a threshold.
			if( fabsf( dd ) < sigma ) {
				float2	dx = make_float2( j - x, i - y );
				AtA.p[0].x += vector1_sq( dx.x );
				AtA.p[1].y += vector1_sq( dx.y );
				AtA.p[1].x += dx.x * dx.y;
				Atb.x += dx.x * dd;
				Atb.y += dx.y * dd;
			}
		}
	}
	AtA.p[0].y = AtA.p[1].x;
	float22	AtA_inv;
	if( float22_inv( &AtA_inv, AtA ) )
		gradient = float22_mv( AtA_inv, Atb );
	gradient_ptr.ptr(y)[x] = gradient;
}

__global__ void d_float_camera_intrinsic_unproject_2( float_camera_intrinsic k, ptr_step_size<float2> image_ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ( x >= image_ptr.cols || y >= image_ptr.rows )
		return;
	image_ptr.ptr(y)[x] = float_camera_intrinsic_unproject_2( k, make_float2( x, y ) );
}

__global__ void d_float_camera_intrinsic_unproject_3( float_camera_intrinsic k, ptr_step_size<float> meter_ptr, ptr_step<float3> vertex_ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ( x >= meter_ptr.cols || y >= meter_ptr.rows )
		return;
	set_vertex_invalid( vertex_ptr.ptr(y)[x] );
	float d = meter_ptr.ptr(y)[x];
	if( d ) vertex_ptr.ptr(y)[x] = float_camera_intrinsic_unproject_3( k, make_float2( x, y ), d );
}

__global__ void d_float_gradient_to_normal( const ptr_step_size<float> meter_ptr, const ptr_step<float2> gradient_ptr,
	const ptr_step<float2> image_ptr, ptr_step<float3> normal_ptr )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ( x >= (meter_ptr.cols-1) || y >= (meter_ptr.rows-1) )
		return;
	float3	normal; set_normal_invalid( normal );
	float	meter = meter_ptr.ptr(y)[x];
	if( is_meter_valid(meter) ) {
		float2	gradient = gradient_ptr.ptr(y)[x];
		float3	vertex_c = float2_make_homogeneous( image_ptr.ptr(y)[x], meter );
		float3	vertex_x = float2_make_homogeneous( image_ptr.ptr(y)[x+1], meter + gradient.x );
		float3	vertex_y = float2_make_homogeneous( image_ptr.ptr(y+1)[x], meter + gradient.y );
		normal = float3_unit( float3_cross( float3_sub(vertex_x, vertex_c), float3_sub(vertex_c, vertex_y) ) );
		//normal = float3_unit( make_float3( gradient.x, gradient.y, -1 ) ); // ?
	}
	normal_ptr.ptr(y)[x] = normal;
}

void cu_kinect_depth_to_vertex( float_camera_intrinsic k, int2 size, short *h_depth, float3 *h_vertex )
{
	cu_array_2d<short>	d_depth( size.y, size.x );
	d_depth.upload( h_depth, size.x * sizeof(short), size.y, size.x );
	cu_array_2d<float>	d_meter( size.y, size.x );
	dim3 block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	dim3 grid = dim3(int_div_up(d_depth.cols(),block.x), int_div_up(d_depth.rows(),block.y));
	d_kinect_depth_to_meter<<<grid,block>>>( d_depth, d_meter );
	cu_sync();
	cu_array_2d<float3>	d_vertex( d_meter.rows(), d_meter.cols() );
	d_float_camera_intrinsic_unproject_3<<<grid,block>>>( k, d_meter, d_vertex );
	cu_sync();
	d_vertex.download( h_vertex, size.x * sizeof(float3) );
}

void d_float_meter_to_normal( float_camera_intrinsic k, const ptr_step_size<float> meter_ptr, ptr_step<float3> normal_ptr )
{
	dim3 block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	dim3 grid = dim3(int_div_up(meter_ptr.cols,block.x), int_div_up(meter_ptr.rows,block.y));
	cu_array_2d<float2>	_gradient( meter_ptr.rows, meter_ptr.cols );
	d_float_meter_to_gradient<<<grid,block>>>( meter_ptr, _gradient, 5, 0.06f );
	cu_sync();
	cu_array_2d<float2>	_image( meter_ptr.rows, meter_ptr.cols );
	d_float_camera_intrinsic_unproject_2<<<grid,block>>>( k, _image );
	cu_sync();
	d_float_gradient_to_normal<<<grid,block>>>( meter_ptr, _gradient, _image, normal_ptr );
	cu_sync();
}

// refrence: Gradient Response Maps for Real-Time Detection of Texture-Less Objects)
void cu_kinect_depth_to_normal( float_camera_intrinsic k, int2 size, short *h_depth, float3 *h_normal )
{
	cu_array_2d<short>	_depth( size.y, size.x );
	_depth.upload( h_depth, size.x * sizeof(short), size.y, size.x );
	cu_array_2d<float>	_meter( size.y, size.x );
	dim3 block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	dim3 grid = dim3(int_div_up(_depth.cols(),block.x), int_div_up(_depth.rows(),block.y));
	d_kinect_depth_to_meter<<<grid,block>>>( _depth, _meter );
	cu_sync();
	cu_array_2d<float3>	_normal( _meter.rows(), _meter.cols() );
	d_float_meter_to_normal( k, _meter, _normal );
	_normal.download( h_normal, size.x * sizeof(float3) );
}

__global__ void d_float_disparity_to_meter( ptr_step_size<float> d_disparity, ptr_step<float> d_meter, float fT, float o )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if( y>=d_disparity.rows || x>=d_disparity.cols )
		return;
	set_meter_invalid( d_meter.ptr(y)[x] );
	if( float1_non_zero( d_disparity.ptr(y)[x] ) ) {
		d_meter.ptr(y)[x] = fT / (d_disparity.ptr(y)[x] + o);
	}
}

void cu_image_disparity_to_normal( float_camera_intrinsic k, float base_line, float disparity_offset,
	int2 size, float *h_disparity, float3 *h_normal )
{
	cu_array_2d<float>	d_disparity( size.y, size.x );
	d_disparity.upload( h_disparity, size.x * sizeof(float), size.y, size.x );
	cu_array_2d<float>	d_meter( size.y, size.x );
	dim3 block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
	dim3 grid = dim3(int_div_up(d_disparity.cols(),block.x), int_div_up(d_disparity.rows(),block.y));
	d_float_disparity_to_meter<<<grid,block>>>( d_disparity, d_meter, k.fc.x * base_line, disparity_offset );
	cu_sync();
	cu_array_2d<float3>	d_normal( d_meter.rows(), d_meter.cols() );
	d_float_meter_to_normal( k, d_meter, d_normal );
	d_normal.download( h_normal, size.x * sizeof(float3) );
}

//////////////////////
// OpenCV dependent //
//////////////////////

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/gpu.hpp>
//#include "cu_matrix.hpp"

extern float_camera_extrinsic float_camera_extrinsic_relative_from_point_matches(
	cv::gpu::GpuMat d_keypoints1, cv::gpu::GpuMat d_descriptors1, cu_array_2d<float3> d_vertex1,
	cv::gpu::GpuMat d_keypoints2, cv::gpu::GpuMat d_descriptors2, cu_array_2d<float3> d_vertex2 );

extern float_camera_extrinsic float_camera_extrinsic_relative_from_point_matches2( 
	float_camera_intrinsic c1_int, cv::gpu::GpuMat d_keypoints1, cv::gpu::GpuMat d_descriptors1, cu_array_2d<float3> d_vertex1,
	float_camera_intrinsic c2_int, cv::gpu::GpuMat d_keypoints2, cv::gpu::GpuMat d_descriptors2, cu_array_2d<float3> d_vertex2, int *inliers_n=0 );

extern float_camera_extrinsic float_camera_extrinsic_relative_from_point_matches3( 
	float_camera_intrinsic c1_int, cv::gpu::GpuMat d_keypoints1, cv::gpu::GpuMat d_descriptors1,
	float_camera_intrinsic c2_int, cv::gpu::GpuMat d_keypoints2, cv::gpu::GpuMat d_descriptors2, cu_array_2d<float3> d_vertex2, int *inliers_n=0 );

extern float_camera_extrinsic float_camera_extrinsic_relative_from_image_pair(
	ptr_step_size<uchar> d_gray1, cu_array_2d<float3> d_vertex1, ptr_step_size<uchar> d_gray2, cu_array_2d<float3> d_vertex2, int *inliers_n=0 );

extern float_camera_extrinsic float_camera_extrinsic_relative_from_image_pair(
	float_camera_intrinsic c1_int, ptr_step_size<uchar> d_gray1, cu_array_2d<float3> d_vertex1,
	float_camera_intrinsic c2_int, ptr_step_size<uchar> d_gray2, cu_array_2d<float3> d_vertex2, int *inliers_n=0 );

// Given:
// kf->h_video and (kf->h_depth or kf->h_disparity)
// Possible options are follows:
// first frame:	true, false, true
// tracking:	true, true, true
// rendering:	false, false, true
// return:		tracking result <-1>:fail, <0>:none, <1>:ICP, <2>:OF, <3>:SURF
int cu_kinect_fusion_process_multimodal( cu_kinect_fusion_params *kf, bool integration, int tracking, bool extraction )
{
	static	int prev_tracking = KINECTFUSION_TRACKING_NONE;

	dim3	block, grid;
	int		trr = KINECTFUSION_TRACKING_NONE; // tracking result

	if( integration || tracking ) {
		if( kf->d_video_dmap ) {
			cu_memcpy_host_to_device( kf->h_video, kf->d_video, kf->video_size_t[0] );
			// undistort
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->video_size[0].x,block.x), int_div_up(kf->video_size[0].y,block.y));
			d_uchar3_mapping<<<grid,block>>>( kf->video_size[0], kf->video_size[0], kf->d_video_dmap, kf->d_video, kf->d_video_u );
			cu_error();
		}
		else {
			cu_memcpy_host_to_device( kf->h_video, kf->d_video_u, kf->video_size_t[0] );
		}
		if( kf->d_depth_dmap ) {
			if( kf->h_disparity ) {
				cu_memcpy_host_to_device( kf->h_disparity, kf->d_disparity, kf->depth_size_t[0] );
				// undistort
				block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
				grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
				d_float_mapping<<<grid,block>>>( kf->depth_size[0], kf->depth_size[0], kf->d_depth_dmap, kf->d_disparity, kf->d_disparity_u );
				cu_error();
			}
			if( kf->h_depth ) {
				cu_memcpy_host_to_device( kf->h_depth, kf->d_depth, kf->depth_size_t[0] );
				// undistort
				block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
				grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
				d_short_mapping<<<grid,block>>>( kf->depth_size[0], kf->depth_size[0], kf->d_depth_dmap, kf->d_depth, kf->d_depth_u );
				cu_error();
			}
		}
		else {
			if( kf->h_disparity )
				cu_memcpy_host_to_device( kf->h_disparity, kf->d_disparity_u, kf->depth_size_t[0] );
			if( kf->h_depth )
				cu_memcpy_host_to_device( kf->h_depth, kf->d_depth_u, kf->depth_size_t[0] );
		}
		cu_sync();
		// undistorted meter
		block = dim3( CU_BLOCK_MAX );
		grid = dim3( int_div_up(kf->depth_size_t[0], block.x) );
		// meter_u[0]
		if( kf->d_disparity_u )
			d_float_disparity_to_meter<<<grid,block>>>( kf->depth_size_t[0], kf->d_disparity_u, kf->d_meter_u[0],
			kf->base_line * kf->depth_int[0].fc.x, kf->disparity_offset );
		else
			d_kinect_depth_to_meter<<<grid,block>>>( kf->depth_size_t[0], kf->d_depth_to_meter, kf->d_depth_u, kf->d_meter_u[0] );
		cu_error(); cu_sync();
	}
	if( tracking ) {
		// common pre-processing of the hybrid approach
		block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
		grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
		// vertex_u[0]
		d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[0], kf->depth_size[0], kf->d_meter_u[0], kf->d_vertex_u[0] );
		cu_error(); cu_sync();
		// vertex_s[0]
		d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[0], kf->depth_size[0], kf->d_meter_s[0], kf->d_vertex_s[0] );
		cu_error(); cu_sync();

		if( tracking == KINECTFUSION_TRACKING_ICP || tracking == KINECTFUSION_TRACKING_FU ) {
			// BEGIN: ICP pre-processing //
			kf->time_buf = nw_tick_count();
			// depth denoising
			cu_memcpy_device_to_device( kf->d_meter_u[0], kf->d_meter_x, kf->depth_size_t[0] );
			_cu_bilateral_filtering( kf->d_meter_x, kf->depth_size[0].x, kf->depth_size[0].y, 3.5f, 0.03f ); // FIXME
	  //_cu_nlm_filtering( kf->d_meter_x, kf->depth_size[0].x, kf->depth_size[0].y, 5.0f, 0.1f ); 

			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[0].x,block.x), int_div_up(kf->depth_size[0].y,block.y));
			// filtered vertex_u[0]
			d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[0], kf->depth_size[0], kf->d_meter_x, kf->d_vertex_x );
			cu_error(); cu_sync();
			// filtered normal_u[0]
			d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[0], kf->d_vertex_x, kf->d_normal_u[0] );
			cu_error(); cu_sync();

			d_float_data_to_mask<<<grid,block>>>( kf->depth_size_t[0], kf->d_meter_x, kf->d_mask[0] );
			cu_error(); cu_sync();
			// meter_u[1]
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[1].x,block.x), int_div_up(kf->depth_size[1].y,block.y));
			d_pyramid_down<<< grid, block>>>( kf->depth_size[0], kf->depth_size[1],
				kf->d_meter_x, kf->d_mask[0], kf->d_meter_u[1], kf->d_mask[1], 0.06f );
			cu_error(); cu_sync();

			block = dim3( CU_BLOCK_MAX );
			grid = dim3( int_div_up(kf->depth_size_t[1], block.x) );
			d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[1], kf->d_mask[1], kf->d_meter_u[1] );
			cu_error(); cu_sync();
			// meter_u[2]
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[2].x,block.x), int_div_up(kf->depth_size[2].y,block.y));
			d_pyramid_down<<< grid, block>>>( kf->depth_size[1], kf->depth_size[2],
				kf->d_meter_u[1], kf->d_mask[1], kf->d_meter_u[2], kf->d_mask[0], 0.06f );
			cu_error(); cu_sync();

			block = dim3( CU_BLOCK_MAX );
			grid = dim3( int_div_up(kf->depth_size_t[2], block.x) );
			d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[2], kf->d_mask[0], kf->d_meter_u[2] );
			cu_error(); cu_sync();

			for( int i=KINECT_DEPTH_LEVEL-1; i>0; i-- ) {
				block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
				grid = dim3(int_div_up(kf->depth_size[i].x,block.x), int_div_up(kf->depth_size[i].y,block.y));
				// vertex_u[i]
				d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[i], kf->depth_size[i], kf->d_meter_u[i], kf->d_vertex_u[i] );
				cu_error(); cu_sync();
				// normal_u[i]
				d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[i], kf->d_vertex_u[i], kf->d_normal_u[i] );
				cu_error(); cu_sync();
			}

			// debug
			//cu_memcpy_device_to_host( kf->d_normal_u[0], (float3*)kf->h_image1, kf->depth_size_t[0] );

			// synthesized meter, vertex, normal (multi-scale)
			block = dim3( CU_BLOCK_MAX );
			grid = dim3( int_div_up(kf->depth_size_t[0], block.x) );
			d_float_data_to_mask<<<grid,block>>>( kf->depth_size_t[0], kf->d_meter_s[0], kf->d_mask[0] );
			cu_error(); cu_sync();

			// meter_s[1]
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[1].x,block.x), int_div_up(kf->depth_size[1].y,block.y));
			d_pyramid_down<<< grid, block>>>( kf->depth_size[0], kf->depth_size[1],
				kf->d_meter_s[0], kf->d_mask[0], kf->d_meter_s[1], kf->d_mask[1], 0.06f );
			cu_error(); cu_sync();

			block = dim3( CU_BLOCK_MAX );
			grid = dim3( int_div_up(kf->depth_size_t[1], block.x) );
			d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[1], kf->d_mask[1], kf->d_meter_s[1] );
			cu_error(); cu_sync();
			// meter_s[2]
			block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
			grid = dim3(int_div_up(kf->depth_size[2].x,block.x), int_div_up(kf->depth_size[2].y,block.y));
			d_pyramid_down<<< grid, block>>>( kf->depth_size[1], kf->depth_size[2],
				kf->d_meter_s[1], kf->d_mask[1], kf->d_meter_s[2], kf->d_mask[0], 0.06f );
			cu_error(); cu_sync();

			block = dim3( CU_BLOCK_MAX );
			grid = dim3( int_div_up(kf->depth_size_t[2], block.x) );
			d_float_mask_to_data<<<grid,block>>>( kf->depth_size_t[2], kf->d_mask[0], kf->d_meter_s[2] );
			cu_error(); cu_sync();

			for( int i=KINECT_DEPTH_LEVEL-1; i>0; i-- ) {
				block = dim3( CU_BLOCK_X, CU_BLOCK_Y );
				grid = dim3(int_div_up(kf->depth_size[i].x,block.x), int_div_up(kf->depth_size[i].y,block.y));
				// vertex_s[i]
				d_img_meter_to_vertex<<<grid,block>>>( kf->depth_int[i], kf->depth_size[i], kf->d_meter_s[i], kf->d_vertex_s[i] );
				cu_error(); cu_sync();
				// normal_s[i]
				d_img_vertex_to_normal<<<grid,block>>>( kf->depth_size[i], kf->d_vertex_s[i], kf->d_normal_s[i] );
				cu_error(); cu_sync();
			}
			// END: ICP pre-processing //

			trr = KINECTFUSION_TRACKING_FAIL;
			const float	distance2_t = 0.001f;
			const float	angle2_t = float1_d2r * 1.f;

			float_camera_extrinsic	relative_ext = float_camera_extrinsic_default();
			// Projection of the current points to the synthesized previous points is good?
			// Preliminary experiments show that the results are almost same
			bool rt = true;
			for( int i=KINECT_DEPTH_LEVEL-1; i>=0; i-- ) {
				kf->icp_params.it_max = kf->icp_it_max[i];
				rt = _cu_icp_projective_3d( kf->icp_params, kf->depth_int[i], &relative_ext, kf->depth_size[i], kf->depth_size[i],
					kf->d_vertex_s[i], kf->d_normal_s[i], kf->d_vertex_u[i], kf->d_normal_u[i], (i)?NULL:kf->d_inlier );
				if( rt == false ) 
				{
					break;
				}
			}
			if( rt ) {
				trr = KINECTFUSION_TRACKING_ICP;
				kf->global_ext = float_camera_extrinsic_compose( relative_ext, kf->global_ext );
				nwf_SO3_coerce( (float*)kf->global_ext.R.p ); // Normalization of Rotation Matrix?
				// Skip the integration process when the current camera motion is very small (optional)
				float	w_[3];	nwf_so3_from_SO3( w_, (const float*)relative_ext.R.p );
				if( float3_norm2( relative_ext.t ) < distance2_t && fabsf(w_[0]) < angle2_t && fabsf(w_[1]) < angle2_t && fabsf(w_[2]) < angle2_t )
					integration = false;
				// debug
				//cu_memcpy_device_to_host( kf->d_inlier, (uchar*)kf->h_image3, kf->depth_size_t[0] );
			}
			kf->time_icp += nw_tick_count() - kf->time_buf; kf->time_count_icp++;
		}

		if( tracking == KINECTFUSION_TRACKING_OF || (tracking == KINECTFUSION_TRACKING_FU && trr == KINECTFUSION_TRACKING_FAIL) ) {
			kf->time_buf = nw_tick_count();

			trr = KINECTFUSION_TRACKING_FAIL;
			const float	distance2_t = 0.001f;
			const float	angle2_t = float1_d2r * 1.f;

			cv::gpu::GpuMat	d_color_u_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC3, kf->d_video_u, kf->video_size[0].x*sizeof(uchar3) );
			cv::gpu::GpuMat	d_gray_u_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC1 );
			cv::gpu::cvtColor( d_color_u_, d_gray_u_, CV_BGR2GRAY );

			cv::gpu::GpuMat	d_color_s_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC3, kf->d_color_s[0], kf->video_size[0].x*sizeof(uchar3) );
			cv::gpu::GpuMat	d_gray_s_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC1 );
			cv::gpu::cvtColor( d_color_s_, d_gray_s_, CV_BGR2GRAY );

#ifdef CG_KLT_TRACKER
			const float distance_t = 0.005f;
			const int inliers_t = 10;
			// Gain-adaptive KLT
			int inliers_n_ = 0;
			float_camera_extrinsic	relative_ext;
			{
#if 0
				// track and search
				if( prev_tracking != KINECTFUSION_TRACKING_OF ) kf->klt.frameNo = 0; // restart tracking
				cv::Mat h_gray_u_;
				d_gray_u_.download( h_gray_u_ );
				kf->klt.track( h_gray_u_.ptr<uchar>() ); // detect, redetect, and track
				int nTracks = kf->klt.get_points();

				cv::Mat h_keypoints1_( 1, nTracks, CV_32FC2, kf->klt.points_curr );
				cv::Mat h_keypoints2_( 1, nTracks, CV_32FC2, kf->klt.points_prev );
				cv::Mat h_status_( 1, nTracks, CV_8UC1 );
				cv::gpu::GpuMat	d_keypoints1_( 1, nTracks, CV_32FC2 );
				cv::gpu::GpuMat	d_keypoints2_( 1, nTracks, CV_32FC2 );
				cv::gpu::GpuMat d_status_( 1, nTracks, CV_8UC1 );
				d_keypoints1_.upload( h_keypoints1_ );
				cv::gpu::PyrLKOpticalFlow d_pyrLK_;
				d_pyrLK_.sparse( d_gray_u_, d_gray_s_, d_keypoints1_, d_keypoints2_, d_status_ );
				d_keypoints2_.download( h_keypoints2_ );
				d_status_.download( h_status_ );

				if( nTracks > inliers_t ) {
					// download vertex
					cu_array_2d<float3>	d_vertex_u( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_u[0], kf->depth_size[0].x*sizeof(float3) );
					cu_array_2d<float3>	d_vertex_s( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_s[0], kf->depth_size[0].x*sizeof(float3) );
					cv::Mat h_vertex_u( d_vertex_u.rows(), d_vertex_u.cols(), CV_32FC3 );
					d_vertex_u.download( h_vertex_u.ptr(), h_vertex_u.step );
					cv::Mat h_vertex_s( d_vertex_s.rows(), d_vertex_s.cols(), CV_32FC3 );
					d_vertex_s.download( h_vertex_s.ptr(), h_vertex_s.step );
					// get matched keypoints
					float2	*h_point2d_1 = alloc_x( float2, nTracks );
					float2	*h_point2d_2 = alloc_x( float2, nTracks );
					float3	*h_point3d_1 = alloc_x( float3, nTracks );
					float3	*h_point3d_2 = alloc_x( float3, nTracks );
					int		nMatches = 0;
					for (int i = 0; i < nTracks; ++i) {
						if( h_status_.ptr<uchar>()[i] == 0 ) continue;
						h_point2d_2[nMatches] = h_keypoints2_.ptr<float2>()[i];
						if( h_point2d_2[nMatches].x < 0.f || h_point2d_2[nMatches].y < 0.f ||
							h_point2d_2[nMatches].x >= kf->video_size[0].x || h_point2d_2[nMatches].y >= kf->video_size[0].y )
							continue;
						h_point3d_2[nMatches] = h_vertex_s.ptr<float3>((int)h_point2d_2[nMatches].y)[(int)h_point2d_2[nMatches].x];
						h_point2d_1[nMatches] = h_keypoints1_.ptr<float2>()[i];
						h_point3d_1[nMatches] = h_vertex_u.ptr<float3>((int)h_point2d_1[nMatches].y)[(int)h_point2d_1[nMatches].x];
						if( is_vertex_valid( h_point3d_1[nMatches] ) && is_vertex_valid( h_point3d_2[nMatches] ) )
							nMatches++;
					} // end for (i)
					if( nMatches > inliers_t ) {
#if 1
						// more stable
						relative_ext = float_camera_extrinsic_relative_from_point_pairs_combined(
							h_point3d_1, h_point3d_2, nMatches, distance_t, &inliers_n_ );
#else
						relative_ext = float_camera_extrinsic_relative_from_point_pairs_drms( h_point3d_1, h_point3d_2, nMatches, distance_t );
						uchar	*h_inliers = alloc_x( uchar, nMatches );
						inliers_n_ = float_camera_extrinsic_relative_get_inliers_crms(
							relative_ext, h_point3d_1, h_point3d_2, h_inliers, nMatches, distance_t );
						free( h_inliers );
#endif
					}
					free( h_point2d_1 );
					free( h_point2d_2 );
					free( h_point3d_1 );
					free( h_point3d_2 );
				}
#elif 1
				// detect and track
				cv::Mat h_gray_u_, h_gray_s_;
				d_gray_u_.download( h_gray_u_ );
				d_gray_s_.download( h_gray_s_ );
				kf->klt.track( h_gray_u_.ptr<uchar>(), 1 ); // detect
				int nTracks = kf->klt.registration( h_gray_s_.ptr<uchar>() ); // track

				cv::Mat h_keypoints1_( 1, nTracks, CV_32FC2, kf->klt.points_prev );
				cv::Mat h_keypoints2_( 1, nTracks, CV_32FC2, kf->klt.points_curr );

				if( nTracks > inliers_t ) {
					// download vertex
					cu_array_2d<float3>	d_vertex_u( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_u[0], kf->depth_size[0].x*sizeof(float3) );
					cu_array_2d<float3>	d_vertex_s( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_s[0], kf->depth_size[0].x*sizeof(float3) );
					cv::Mat h_vertex_u( d_vertex_u.rows(), d_vertex_u.cols(), CV_32FC3 );
					d_vertex_u.download( h_vertex_u.ptr(), h_vertex_u.step );
					cv::Mat h_vertex_s( d_vertex_s.rows(), d_vertex_s.cols(), CV_32FC3 );
					d_vertex_s.download( h_vertex_s.ptr(), h_vertex_s.step );
					// get matched keypoints
					float2	*h_point2d_1 = alloc_x( float2, nTracks );
					float2	*h_point2d_2 = alloc_x( float2, nTracks );
					float3	*h_point3d_1 = alloc_x( float3, nTracks );
					float3	*h_point3d_2 = alloc_x( float3, nTracks );
					int		nMatches = 0;
					for (int i = 0; i < nTracks; ++i) {
						h_point2d_2[nMatches] = h_keypoints2_.ptr<float2>()[i];
						if( h_point2d_2[nMatches].x < 0.f || h_point2d_2[nMatches].y < 0.f ||
							h_point2d_2[nMatches].x >= kf->video_size[0].x || h_point2d_2[nMatches].y >= kf->video_size[0].y )
							continue;
						h_point3d_2[nMatches] = h_vertex_s.ptr<float3>((int)h_point2d_2[nMatches].y)[(int)h_point2d_2[nMatches].x];
						h_point2d_1[nMatches] = h_keypoints1_.ptr<float2>()[i];
						h_point3d_1[nMatches] = h_vertex_u.ptr<float3>((int)h_point2d_1[nMatches].y)[(int)h_point2d_1[nMatches].x];
						if( is_vertex_valid( h_point3d_1[nMatches] ) && is_vertex_valid( h_point3d_2[nMatches] ) )
							nMatches++;
					} // end for (i)

					if( nMatches > inliers_t ) {
						relative_ext = float_camera_extrinsic_relative_from_point_pairs_combined(
							h_point3d_1, h_point3d_2, nMatches, distance_t, &inliers_n_ );
					}
					free( h_point2d_1 );
					free( h_point2d_2 );
					free( h_point3d_1 );
					free( h_point3d_2 );
				}
#else
				// two steps: track and search. if failed, detect and track.
				// download //
				cv::Mat h_gray_u_;
				d_gray_u_.download( h_gray_u_ );
				// download vertex
				cu_array_2d<float3>	d_vertex_u( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_u[0], kf->depth_size[0].x*sizeof(float3) );
				cu_array_2d<float3>	d_vertex_s( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_s[0], kf->depth_size[0].x*sizeof(float3) );
				cv::Mat h_vertex_u( d_vertex_u.rows(), d_vertex_u.cols(), CV_32FC3 );
				d_vertex_u.download( h_vertex_u.ptr(), h_vertex_u.step );
				cv::Mat h_vertex_s( d_vertex_s.rows(), d_vertex_s.cols(), CV_32FC3 );
				d_vertex_s.download( h_vertex_s.ptr(), h_vertex_s.step );
				// first : track and search
				{
					if( prev_tracking != KINECTFUSION_TRACKING_OF ) kf->klt.frameNo = 0; // restart tracking
					kf->klt.track( h_gray_u_.ptr<uchar>() ); // detect, redetect, and track
					int nTracks = kf->klt.get_points();

					cv::Mat h_keypoints1_( 1, nTracks, CV_32FC2, kf->klt.points_curr );
					cv::Mat h_keypoints2_( 1, nTracks, CV_32FC2, kf->klt.points_prev );
					cv::Mat h_status_( 1, nTracks, CV_8UC1 );
					cv::gpu::GpuMat	d_keypoints1_( 1, nTracks, CV_32FC2 );
					cv::gpu::GpuMat	d_keypoints2_( 1, nTracks, CV_32FC2 );
					cv::gpu::GpuMat d_status_( 1, nTracks, CV_8UC1 );
					d_keypoints1_.upload( h_keypoints1_ );
					cv::gpu::PyrLKOpticalFlow d_pyrLK_;
					d_pyrLK_.sparse( d_gray_u_, d_gray_s_, d_keypoints1_, d_keypoints2_, d_status_ );
					d_keypoints2_.download( h_keypoints2_ );
					d_status_.download( h_status_ );

					if( nTracks > inliers_t ) {
						// get matched keypoints
						float2	*h_point2d_1 = alloc_x( float2, nTracks );
						float2	*h_point2d_2 = alloc_x( float2, nTracks );
						float3	*h_point3d_1 = alloc_x( float3, nTracks );
						float3	*h_point3d_2 = alloc_x( float3, nTracks );
						int		nMatches = 0;
						for (int i = 0; i < nTracks; ++i) {
							if( h_status_.ptr<uchar>()[i] == 0 ) continue;
							h_point2d_2[nMatches] = h_keypoints2_.ptr<float2>()[i];
							if( h_point2d_2[nMatches].x < 0.f || h_point2d_2[nMatches].y < 0.f ||
								h_point2d_2[nMatches].x >= kf->video_size[0].x || h_point2d_2[nMatches].y >= kf->video_size[0].y )
								continue;
							h_point3d_2[nMatches] = h_vertex_s.ptr<float3>((int)h_point2d_2[nMatches].y)[(int)h_point2d_2[nMatches].x];
							h_point2d_1[nMatches] = h_keypoints1_.ptr<float2>()[i];
							h_point3d_1[nMatches] = h_vertex_u.ptr<float3>((int)h_point2d_1[nMatches].y)[(int)h_point2d_1[nMatches].x];
							if( is_vertex_valid( h_point3d_1[nMatches] ) && is_vertex_valid( h_point3d_2[nMatches] ) )
								nMatches++;
						} // end for (i)
						if( nMatches > inliers_t ) {
							relative_ext = float_camera_extrinsic_relative_from_point_pairs_combined(
								h_point3d_1, h_point3d_2, nMatches, distance_t, &inliers_n_ );
						}
						free( h_point2d_1 );
						free( h_point2d_2 );
						free( h_point3d_1 );
						free( h_point3d_2 );
					}
				}
				// second step: detect and track
				if( inliers_n_ <= inliers_t ) {
					cv::Mat h_gray_s_;
					d_gray_s_.download( h_gray_s_ );
					// no track
					kf->klt.track( h_gray_u_.ptr<uchar>(), 1 ); // detect
					int nTracks = kf->klt.registration( h_gray_s_.ptr<uchar>() ); // track
					kf->klt.frameNo = 0;

					cv::Mat h_keypoints1_( 1, nTracks, CV_32FC2, kf->klt.points_prev );
					cv::Mat h_keypoints2_( 1, nTracks, CV_32FC2, kf->klt.points_curr );

					if( nTracks > inliers_t ) {
						// get matched keypoints
						float2	*h_point2d_1 = alloc_x( float2, nTracks );
						float2	*h_point2d_2 = alloc_x( float2, nTracks );
						float3	*h_point3d_1 = alloc_x( float3, nTracks );
						float3	*h_point3d_2 = alloc_x( float3, nTracks );
						int		nMatches = 0;
						for (int i = 0; i < nTracks; ++i) {
							h_point2d_2[nMatches] = h_keypoints2_.ptr<float2>()[i];
							if( h_point2d_2[nMatches].x < 0.f || h_point2d_2[nMatches].y < 0.f ||
								h_point2d_2[nMatches].x >= kf->video_size[0].x || h_point2d_2[nMatches].y >= kf->video_size[0].y )
								continue;
							h_point3d_2[nMatches] = h_vertex_s.ptr<float3>((int)h_point2d_2[nMatches].y)[(int)h_point2d_2[nMatches].x];
							h_point2d_1[nMatches] = h_keypoints1_.ptr<float2>()[i];
							h_point3d_1[nMatches] = h_vertex_u.ptr<float3>((int)h_point2d_1[nMatches].y)[(int)h_point2d_1[nMatches].x];
							if( is_vertex_valid( h_point3d_1[nMatches] ) && is_vertex_valid( h_point3d_2[nMatches] ) )
								nMatches++;
						} // end for (i)

						if( nMatches > inliers_t ) {
							relative_ext = float_camera_extrinsic_relative_from_point_pairs_combined(
								h_point3d_1, h_point3d_2, nMatches, distance_t, &inliers_n_ );
						}
						free( h_point2d_1 );
						free( h_point2d_2 );
						free( h_point3d_1 );
						free( h_point3d_2 );
					}
				}
#endif
			}
#else
			// Good feature to track + optical flow
			const int inliers_t = 30;
			// OpenCV
			cu_array_2d<uchar>	d_gray_u( kf->depth_size[0].y, kf->depth_size[0].x, d_gray_u_.ptr(), d_gray_u_.step );
			cu_array_2d<uchar>	d_gray_s( kf->depth_size[0].y, kf->depth_size[0].x, d_gray_s_.ptr(), d_gray_s_.step );
			cu_array_2d<float3>	d_vertex_u( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_u[0], kf->depth_size[0].x*sizeof(float3) );
			cu_array_2d<float3>	d_vertex_s( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_s[0], kf->depth_size[0].x*sizeof(float3) );

			int inliers_n_;
#if 0
			float_camera_extrinsic	relative_ext = float_camera_extrinsic_relative_from_image_pair(
				kf->video_int[0], d_gray_u, d_vertex_u, kf->video_int[0], d_gray_s, d_vertex_s, &inliers_n_ );
#else
			float_camera_extrinsic	relative_ext = float_camera_extrinsic_relative_from_image_pair(
				d_gray_u, d_vertex_u, d_gray_s, d_vertex_s, &inliers_n_ );
#endif
#endif
			if( inliers_n_ > inliers_t ) {
				const float	distance1_t = 0.2f;
				const float	angle1_t = float1_d2r * 15.f;

				float	w_[3];	nwf_so3_from_SO3( w_, (const float*)relative_ext.R.p );
				// out of bound (200 mm, 15 degree)
				if( float3_norm2( relative_ext.t ) > distance1_t || fabsf(w_[0]) > angle1_t || fabsf(w_[1]) > angle1_t || fabsf(w_[2]) > angle1_t ) {
					nw_cui_error( "out of bound - R=[%f,%f,%f], t=[%f,%f,%f]\n", w_[0]*float1_r2d, w_[1]*float1_r2d, w_[2]*float1_r2d,
						relative_ext.t.x, relative_ext.t.y, relative_ext.t.z );
				}
				else {
					trr = KINECTFUSION_TRACKING_OF;
					kf->global_ext = float_camera_extrinsic_compose( relative_ext, kf->global_ext );
					nwf_SO3_coerce( (float*)kf->global_ext.R.p ); // Normalization of Rotation Matrix?
					// Skip the integration process when the current camera motion is very small (optional)
					if( float3_norm2( relative_ext.t ) < distance2_t && fabsf(w_[0]) < angle2_t && fabsf(w_[1]) < angle2_t && fabsf(w_[2]) < angle2_t )
						integration = false;
				}
			}
			//else {
			//	nw_cui_debug( "fail to track." );
			//}
			//nw_cui_debug( "OF inliers = %d", inliers_n_ );
			kf->time_of += nw_tick_count() - kf->time_buf; kf->time_count_of++;
		}
		if( tracking == KINECTFUSION_TRACKING_SURF/* || (tracking == KINECTFUSION_TRACKING_FU && trr == KINECTFUSION_TRACKING_FAIL)*/ ) {
			kf->time_buf = nw_tick_count();

			trr = KINECTFUSION_TRACKING_FAIL;
			const float	distance2_t = 0.001f;
			const float	angle2_t = float1_d2r * 1.f;

			cv::gpu::GpuMat	d_color_u_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC3, kf->d_video_u, kf->video_size[0].x*sizeof(uchar3) );
			cv::gpu::GpuMat	d_gray_u_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC1 );
			cv::gpu::cvtColor( d_color_u_, d_gray_u_, CV_BGR2GRAY );

			cv::gpu::GpuMat	d_color_s_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC3, kf->d_color_s[0], kf->video_size[0].x*sizeof(uchar3) );
			cv::gpu::GpuMat	d_gray_s_( kf->video_size[0].y, kf->video_size[0].x, CV_8UC1 );
			cv::gpu::cvtColor( d_color_s_, d_gray_s_, CV_BGR2GRAY );

			cv::gpu::SURF_GPU surf_( 100., 2, 4 );
			cv::gpu::GpuMat	d_keypoints_u_, d_keypoints_s_;
			cv::gpu::GpuMat	d_descriptors_u_, d_descriptors_s_;
			surf_( d_gray_u_, cv::gpu::GpuMat(), d_keypoints_u_, d_descriptors_u_ );
			surf_( d_gray_s_, cv::gpu::GpuMat(), d_keypoints_s_, d_descriptors_s_ );

			if( d_keypoints_u_.cols > 3 && d_keypoints_s_.cols > 3 ) {
				cu_array_2d<float3>	d_vertex_u_( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_u[0], kf->depth_size[0].x*sizeof(float3) );
				cu_array_2d<float3>	d_vertex_s_( kf->depth_size[0].y, kf->depth_size[0].x, kf->d_vertex_s[0], kf->depth_size[0].x*sizeof(float3) );

				int inliers_n_;
				//float_camera_extrinsic	relative_ext = float_camera_extrinsic_relative_from_point_matches3(
				//	kf->video_int[0], d_keypoints_u_, d_descriptors_u_,
				//	kf->video_int[0], d_keypoints_s_, d_descriptors_s_, d_vertex_s_, &inliers_n_ );				
		float_camera_extrinsic	relative_ext = float_camera_extrinsic_relative_from_point_matches2(
					kf->video_int[0], d_keypoints_u_, d_descriptors_u_, d_vertex_u_,
					kf->video_int[0], d_keypoints_s_, d_descriptors_s_, d_vertex_s_, &inliers_n_ );
				if( inliers_n_ > 30 ) {
					trr = KINECTFUSION_TRACKING_SURF;
					float	w_[3];	nwf_so3_from_SO3( w_, (const float*)relative_ext.R.p );
					kf->global_ext = float_camera_extrinsic_compose( relative_ext, kf->global_ext );
					nwf_SO3_coerce( (float*)kf->global_ext.R.p ); // Normalization of Rotation Matrix?
					// Skip the integration process when the current camera motion is very small (optional)
					if( float3_norm2( relative_ext.t ) < distance2_t && fabsf(w_[0]) < angle2_t && fabsf(w_[1]) < angle2_t && fabsf(w_[2]) < angle2_t )
						integration = false;
				}
			}
			kf->time_surf += nw_tick_count() - kf->time_buf; kf->time_count_surf++;
		}
	}
	//printf("integration: %d  tracking mode: %d\n",integration,trr);
	if( integration && (trr==KINECTFUSION_TRACKING_ICP || trr==KINECTFUSION_TRACKING_OF || trr==KINECTFUSION_TRACKING_NONE) ) {
		kf->time_buf = nw_tick_count();
		// Note that severe drifting errors are occured when the filtered depth map is used
		_cu_voxel_volumetric_integration( kf->volume, kf->depth_int[0], kf->global_ext, kf->depth_size[0], kf->d_meter_u[0],
			(tracking == KINECTFUSION_TRACKING_ICP || tracking == KINECTFUSION_TRACKING_FU)?kf->d_normal_u[0]:NULL, kf->d_video_u, false );
		kf->time_integration += nw_tick_count() - kf->time_buf; kf->time_count_integration++;
	}
	if( extraction ) {
		//printf("extraction\n");
		kf->time_buf = nw_tick_count();
		// meter_s[0], normal_s[0]
		_cu_voxel_extract_depth_and_normal_map( kf->volume, kf->depth_int[0], kf->global_ext,
			kf->depth_size[0], kf->d_meter_s[0], kf->d_normal_s[0], kf->d_color_s[0] );
		// debug
		//cu_memcpy_device_to_host( kf->d_normal_s[0], (float3*)kf->h_image2, kf->depth_size_t[0] );
		kf->time_extraction += nw_tick_count() - kf->time_buf; kf->time_count_extraction++;
	}

	// Experiment: frame-to-frame color tracking -> FAIL to track :-)
#if 0
	cu_memcpy_device_to_device( kf->d_video_u, kf->d_color_s[0], kf->video_size_t[0] );
#endif

	kf->time_count++;
	prev_tracking = trr;
	return trr;
}

#endif

#include "cu_marching_cubes_v1.h"

void read_volume_from_file ( char * file_, cu_voxel& local_volume_)
{

}

__device__ float3 _d_voxel_extract_surface_normal( int width, cu_voxel_t *data1_ptr, cu_voxel_t *data2_ptr, float3 p_f, int3 p_i )
{
	cu_voxel_t	*data_ptr;
	short		sdf_i;
	float3		t_f, n_f;
	float		sdfs[8], vxy[4], sdf1, sdf2;
	int	index = p_i.y * width + p_i.x;
	vector3_set( n_f, 0.0f );
	vector3_sub( t_f, p_f, p_i );

	data_ptr = data1_ptr + index;
	sdf_i = data_ptr->tsdf;	// V000
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[0] = d_unpack_tsdf( sdf_i );

	data_ptr += 1;
	sdf_i = data_ptr->tsdf;	// V100
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[1] = d_unpack_tsdf( sdf_i );

	data_ptr += width;
	sdf_i = data_ptr->tsdf;	// V110
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[6] = d_unpack_tsdf( sdf_i );

	data_ptr -= 1;
	sdf_i = data_ptr->tsdf;	// V010
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[2] = d_unpack_tsdf( sdf_i );

	data_ptr = data2_ptr + index;
	sdf_i = data_ptr->tsdf;	// V001
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[3] = d_unpack_tsdf( sdf_i );

	data_ptr += 1;
	sdf_i = data_ptr->tsdf;	// V101
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[4] = d_unpack_tsdf( sdf_i );

	data_ptr += width;
	sdf_i = data_ptr->tsdf;	// V111
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[7] = d_unpack_tsdf( sdf_i );

	data_ptr -= 1;
	sdf_i = data_ptr->tsdf;	// V011
	if( sdf_i == CU_VOXEL_MAX_TSDF || sdf_i == -CU_VOXEL_MAX_TSDF )	return n_f;
	sdfs[5] = d_unpack_tsdf( sdf_i );

#if 1
	// x component
	vxy[0] = sdfs[0]; vxy[1] = sdfs[2]; vxy[2] = sdfs[3]; vxy[3] = sdfs[5];
	sdf1 = float_interpolation_bilinear( vxy, t_f.y, t_f.z );
	vxy[0] = sdfs[1]; vxy[1] = sdfs[6]; vxy[2] = sdfs[4]; vxy[3] = sdfs[7];
	sdf2 = float_interpolation_bilinear( vxy, t_f.y, t_f.z );
	n_f.x = sdf2 - sdf1;
	// y component
	vxy[0] = sdfs[0]; vxy[1] = sdfs[1]; vxy[2] = sdfs[3]; vxy[3] = sdfs[4];
	sdf1 = float_interpolation_bilinear( vxy, t_f.x, t_f.z );
	vxy[0] = sdfs[2]; vxy[1] = sdfs[6]; vxy[2] = sdfs[5]; vxy[3] = sdfs[7];
	sdf2 = float_interpolation_bilinear( vxy, t_f.x, t_f.z );
	n_f.y = sdf2 - sdf1;
	// z component
	vxy[0] = sdfs[0]; vxy[1] = sdfs[1]; vxy[2] = sdfs[2]; vxy[3] = sdfs[6];
	sdf1 = float_interpolation_bilinear( vxy, t_f.x, t_f.y );
	vxy[0] = sdfs[3]; vxy[1] = sdfs[4]; vxy[2] = sdfs[5]; vxy[3] = sdfs[7];
	sdf2 = float_interpolation_bilinear( vxy, t_f.x, t_f.y );
	n_f.z = sdf2 - sdf1;
#else
	// Significantly degrade the performance!
	// The piecewise constant gradient (using a constant average of finite differences discretization)
	// Reference: Eq.11 of F. Calakli and G. Taubin, "Smooth Signed Distance Surface Reconstruction"
	n_f.x = sdfs[1] - sdfs[0] + sdfs[4] - sdfs[3] + sdfs[6] - sdfs[2] + sdfs[7] - sdfs[5];
	n_f.y = sdfs[2] - sdfs[0] + sdfs[5] - sdfs[3] + sdfs[6] - sdfs[1] + sdfs[7] - sdfs[4];
	n_f.z = sdfs[3] - sdfs[0] + sdfs[5] - sdfs[2] + sdfs[4] - sdfs[1] + sdfs[7] - sdfs[6];
//	return float3_div_c( n_f, 4.0 * 0.003f ); // FIXME
#endif
	return float3_unit( n_f );
}

__device__ float3 d_voxel_extract_surface_normal( cu_voxel *v, float3 p_f )
{
	int3 p_i;
	vector3_copy( p_i, p_f );
	cu_voxel_t	*data_ptr = cu_voxel_access_z( v, p_i.z );
	return _d_voxel_extract_surface_normal( v->size.x, data_ptr, data_ptr + v->size_xy, p_f, p_i );
}


__global__ void d_voxel_extract_surface_normals( cu_voxel v, int size, vertex_voxel *vertex_ptr, float3 *normal_ptr )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i >= size )
		return;
	normal_ptr[i] = d_voxel_extract_surface_normal( &v, float3_div_c( vertex_ptr[i].ver_pos, v.grid_s ) ); // FIXME
}


void cu_voxel_delete( cu_voxel *v )
{
	if( v->data ) {
		cu_free( v->data );
		v->data		= NULL;
		v->size.x	= 0;
		v->size.y	= 0;
		v->size.z	= 0;
		v->min_t	= 0.0f;
		v->max_t	= 0.0f;
		v->grid_s	= 0.0f;
		checkCudaErrors( cudaGetLastError() ); cu_sync();
	}
}


void cu_voxel_rlc_to_mesh( char *file_input, char *file_output, int volume_dim)
{
	cu_voxel	volume; volume.data = NULL;
	cu_voxel	local_volume; local_volume.data = NULL;
	//cu_voxel_run_length_decode( &volume, file_input, make_int3(0,0,0), make_int3(0,0,0), make_int3(0,0,0) );


	//---------------------------------------------------------------------------------------------------------
	//modify this value
	local_volume.grid_s = 0.003;
	local_volume.size.x = volume_dim;
	local_volume.size.y = volume_dim;
	local_volume.size.z = volume_dim;
	local_volume.size_xy = local_volume.size.x * local_volume.size.y;
	local_volume.data = new cu_voxel_t[local_volume.size.x*local_volume.size.y*local_volume.size.z];
	//----------------------------------------------------------------------------------------------------------
	volume.grid_s = local_volume.grid_s;
	volume.size.x = local_volume.size.x;
	volume.size.y = local_volume.size.y;
	volume.size.z = local_volume.size.z;
	volume.size_xy = local_volume.size_xy;
	checkCudaErrors(cudaMalloc( &volume.data, volume.size_xy * volume.size.z * sizeof(cu_voxel_t) )); 
	
	//----------------------------------------------------------------------------------------------------------------
	// file open
	std::ifstream ifs;
	ifs.open(file_input, std::ios_base::in);

	if(ifs.is_open()==false)
	{
		printf("fail to open %s",file_input);
		return;
	}
	
	char data[256];

	//skip first data
	ifs.getline(data,256);

	int x_count = 0;
	int y_count = 0;
	int z_count = 0;
	int sdf = 0;

	int valid_count = 0;
	
	while(ifs.eof() == false )
	{
		ifs.getline(data,256);
		int token_id = 0;
		char* token;
		token = strtok(data," ");	//split data using space bar
		while(token!= NULL)
		{
			if( token_id == 0 )
			{
				x_count = atoi(token);
				token_id++;
			}
			else if (token_id == 1 )
			{
				y_count = atoi(token);
				token_id++;
			}
			else if (token_id == 2 )
			{
				z_count = atoi(token);
				token_id++;
			}
			else if (token_id == 3 )
			{
				sdf = atoi(token);
				
				token_id++;
			}
			else if ( token_id == 4 )
			{
				local_volume.data[z_count * volume.size_xy + y_count * volume.size.x + x_count].w = atoi(token);
				local_volume.data[z_count * volume.size_xy + y_count * volume.size.x + x_count].tsdf = sdf;
			}

			if(token_id!=4)
			{
				token = strtok(NULL, " ");
			}
			else
			{
				token = strtok(NULL, "\n");
			}
		}
		
	}
	checkCudaErrors(cudaMemcpy(volume.data, local_volume.data, volume.size_xy*volume.size.z * sizeof(cu_voxel_t),cudaMemcpyHostToDevice));
	
	nw_cui_message_s( "cu_voxel_rlc_to_mesh - generating triangles..." );
	MarchingCubes mc;
	cu_array<vertex_voxel> triangles_buffer; // If a memory error returns, adjust DEFAULT_TRIANGLES_BUFFER_SIZE!
	cu_array<int3> vertexindex_to_voxel;
	cu_array<short> ver_voxel;
	cu_array<vertex_voxel> triangles_device = mc.run(volume, triangles_buffer, vertexindex_to_voxel, ver_voxel/*inyeop*/);
	nw_cui_message_e( "done" );
	nw_cui_message( "cu_voxel_rlc_to_mesh - # of triangles = %d", triangles_device.size()/3 );

	nw_cui_message_s( "cu_voxel_rlc_to_mesh - generating normals..." );
	cu_array<float3>	normals_buffer(triangles_device.size());
	dim3	block = dim3( CU_BLOCK_MAX );
	dim3	grid = dim3( int_div_up(normals_buffer.size(), block.x) );
	d_voxel_extract_surface_normals <<< grid,block >>> ( volume, normals_buffer.size(), triangles_device.ptr(), normals_buffer.ptr() );
	nw_cui_message_e( "done" );

//	cu_voxel_delete( &volume );

	vertex_voxel	*h_vertices = alloc_x( vertex_voxel, triangles_device.size() );
	triangles_device.download( h_vertices );

	float3	*h_normals = alloc_x( float3, normals_buffer.size() );
	normals_buffer.download( h_normals );

	int3	*h_vertexindex_to_voxel = alloc_x(int3, vertexindex_to_voxel.size());
	vertexindex_to_voxel.download(h_vertexindex_to_voxel);

	short* h_ver_voxel = alloc_x(short, ver_voxel.size());
	ver_voxel.download(h_ver_voxel);
	
  char *ext_ = strrchr(file_output,'.');
	if( strcmp( ext_, ".ply" ) == 0 ) {
		// write to ply file
		nw_cui_message_s( "cu_voxel_rlc_to_mesh - write to a ply file..." );
		FILE *fp = fopen( file_output, "wb" );
		fprintf( fp, "ply\n" );
		fprintf( fp, "format binary_little_endian 1.0\n" );
		fprintf( fp, "element vertex %d\n", triangles_device.size() );
		fprintf( fp, "property float32 x\n" );
		fprintf( fp, "property float32 y\n" );
		fprintf( fp, "property float32 z\n" );
		fprintf( fp, "property float32 nx\n" );
		fprintf( fp, "property float32 ny\n" );
		fprintf( fp, "property float32 nz\n" );
		fprintf( fp, "element face %d\n", triangles_device.size()/3 );
		fprintf( fp, "property list uint8 int32 vertex_indices\n" );
		fprintf( fp, "end_header\n" );
		for( int i=0; i<triangles_device.size(); i++ ) {
			fwrite( &(h_vertices+i)->ver_pos, sizeof(float3), 1, fp );
			fwrite( h_normals+i, sizeof(float3), 1, fp );
		}
		uchar	prop_ = 3;
		int		face_[3];
		for( int i=0; i<triangles_device.size()/3; i++ ) {
			face_[0] = i*3; face_[1] = face_[0]+1; face_[2] = face_[1]+1;
			fwrite( &prop_, 1, 1, fp );
			fwrite( face_, sizeof(int), 3, fp );
		}
		fclose( fp );
		nw_cui_message_e( "done" );
	}
	else if( strcmp( ext_, ".obj" ) == 0 ) {
		// write to obj file
		nw_cui_message_s( "cu_voxel_rlc_to_mesh - write to an obj file..." );
		FILE *fp = fopen( file_output, "w" );
		for( int i=0; i<triangles_device.size(); i++ ) {
			fprintf( fp, "v %f %f %f\n", h_vertices[i].ver_pos.x, h_vertices[i].ver_pos.y, h_vertices[i].ver_pos.z );
			fprintf( fp, "vn %f %f %f\n", h_normals[i].x, h_normals[i].y, h_normals[i].z );
		}
		for( int i=0; i<triangles_device.size()/3; i++ ) {
			fprintf( fp, "f %d %d %d\n", i*3+1, i*3+2, i*3+3 );
		}
		fclose( fp );
		nw_cui_message_e( "done" );
	}
	else {
		nw_cui_error( "cu_voxel_rlc_to_mesh - Invalid file extension (%s)", ext_ );
	}
	if ( h_vertices != NULL)
	{
		free( h_vertices );
	}
	if ( h_normals != NULL)
	{
		free( h_normals );
	}
	
}

/*__global__ void AssignVolume(short2* fromhostData, cu_voxel_t* todevData, int numElements)
{
	int tx= blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;

	int tid = 256 * ty + tx;

	if (tid < numElements)
	{
		todevData[tid].tsdf = fromhostData[tid].x;
		todevData[tid].w = fromhostData[tid].y;
	}

	
}
*/
/*__global__ void AssignVolume(short2* fromhostData, cu_voxel* v)
{
	
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int z = threadIdx.y + blockIdx.y * blockDim.y;

	int index = z*v->size.x + y;
	if (y >= v->size.y || z >= v->size.z)
		return;

	fromhostData=fromhostData + (z)* (v)->size_xy + y*v->size.x; /// [z][y][]


	cu_voxel_t	*ptr_s = cu_voxel_access_yz(v, y, z);
	cu_voxel_t	*ptr_e = ptr_s + v->size.x;
	do {
		ptr_s->tsdf = fromhostData->x;
		ptr_s->w = fromhostData->y;
		ptr_s->color.x = 0;
		ptr_s->color.y = 0;
		ptr_s->color.z = 0;
		++fromhostData;
	} while ((++ptr_s) != ptr_e);
}
*/

#include "../vector_types.h"
#include <time.h>

inline __device__ uint2 thr2pos2() {
#ifdef __CUDACC__
	return make_uint2(__umul24(blockDim.x, blockIdx.x) + threadIdx.x,
		__umul24(blockDim.y, blockIdx.y) + threadIdx.y);
#else
	return make_uint2(0);
#endif
}


__global__ void AssignVolume(short2* fromhostData, int3 size, cu_voxel_t* todevData)
{
	uint3 pos = make_uint3(thr2pos2());
	for (pos.z = 0; pos.z < 256; ++pos.z)
	{
		todevData[pos.x + pos.y * size.x + pos.z * size.x * size.y].tsdf = fromhostData[pos.x + pos.y * size.x + pos.z * size.x * size.y].x;
		todevData[pos.x + pos.y * size.x + pos.z * size.x * size.y].w = fromhostData[pos.x + pos.y * size.x + pos.z * size.x * size.y].y;
	}
}

inline int divup(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
inline dim3 divup(uint2 a, dim3 b) {
	return dim3(divup(a.x, b.x), divup(a.y, b.y));
}
inline dim3 divup(dim3 a, dim3 b) {
	return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z));
}
void cu_voxel_to_mesh(short2* _local_volume_data, unsigned int _size_x, unsigned int _size_y, unsigned int _size_z, float _grid_s, const char *file_output)
{
	

	int numElements = _size_x*_size_y*_size_z;

	
//	int threadsPerBlock = 256;
//	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;


	//dim3	grid1 = dim3(int_div_up(numElements, block1.x));
	


	clock_t begin;
	begin = clock();

	cu_voxel volume; volume.data = NULL;
	volume.grid_s = _grid_s;
	volume.size.x = _size_x;	volume.size.y = _size_y;	volume.size.z =_size_z;
	volume.size_xy = _size_x*_size_y;
	//volume.data = (cu_voxel_t*)_local_volume_data;
	checkCudaErrors(cudaMalloc((void**)&volume.data, volume.size_xy * volume.size.z * sizeof(cu_voxel_t)));
	//local_volume->data = new cu_voxel_t[local_volume->size.x*local_volume->size.y*local_volume->size.z];



/*	dim3	block1 = dim3(_size_x, 1, 1);
	dim3	grid1(_size_y, _size_z, 1);
	AssignVolume << <grid1, block1 >> >(_local_volume_data, volume.data, numElements);
	*/

	//dim3 block1(CU_BLOCK_X, CU_BLOCK_Y);
	//dim3 grid1(int_div_up(volume.size.y, block1.x), int_div_up(volume.size.z, block1.y));
	

	dim3 block1(32, 16);
	dim3 grid1 = divup(dim3(volume.size.x, volume.size.y), block1);

	AssignVolume <<<grid1, block1 >>>(_local_volume_data, volume.size, volume.data);


	cu_voxel_t* h_volume;
	h_volume = (cu_voxel_t*)malloc(_size_x*_size_y*_size_z*sizeof(cu_voxel_t));
	cudaError_t err = cudaMemcpy(h_volume, volume.data, _size_x*_size_y*_size_z*sizeof(cu_voxel_t), cudaMemcpyDeviceToHost);


	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	/*FILE* fp;
	fp = fopen("test.txt", "wt");
	for (int i = 0; i < _size_x*_size_y*_size_z; i++)
		fprintf(fp, "%u %u\n", (h_volume + i)->tsdf, (h_volume + i)->w);
	fclose(fp);
	*/

	//checkCudaErrors(cudaMemcpy(volume.data, _local_volume_data, volume.size_xy*volume.size.z * sizeof(cu_voxel_t), cudaMemcpyHostToDevice));
	
	nw_cui_message_s("cu_voxel_to_mesh - generating triangles...");
	MarchingCubes mc;
	cu_array<vertex_voxel> triangles_buffer; // If a memory error returns, adjust DEFAULT_TRIANGLES_BUFFER_SIZE!
	cu_array<int3> vertexindex_to_voxel;
	cu_array<short> ver_voxel;

	cu_array<vertex_voxel> cu_v_v;

	cu_array<vertex_voxel> triangles_device = mc.run(volume, triangles_buffer, vertexindex_to_voxel, ver_voxel/*inyeop*/);
	nw_cui_message_e("done");
	nw_cui_message("cu_voxel_to_mesh - # of triangles = %d", triangles_device.size() / 3);

	nw_cui_message_s("cu_voxel_to_mesh - generating normals...");
	cu_array<float3>	normals_buffer(triangles_device.size());
	dim3	block = dim3(CU_BLOCK_MAX);
	dim3	grid = dim3(int_div_up(normals_buffer.size(), block.x));
	d_voxel_extract_surface_normals << <grid, block >> >(volume, normals_buffer.size(), triangles_device.ptr(), normals_buffer.ptr());
	nw_cui_message_e("done");

	cu_voxel_delete(&volume);

	vertex_voxel	*h_vertices = alloc_x(vertex_voxel, triangles_device.size());
	triangles_device.download(h_vertices);

	float3	*h_normals = alloc_x(float3, normals_buffer.size());
	normals_buffer.download(h_normals);

	int3	*h_vertexindex_to_voxel = alloc_x(int3, vertexindex_to_voxel.size());
	vertexindex_to_voxel.download(h_vertexindex_to_voxel);

	short* h_ver_voxel = alloc_x(short, ver_voxel.size());
	//memset(h_ver_voxel, 32766, ver_voxel.size()*sizeof(short));
	ver_voxel.download(h_ver_voxel);

	ver_voxel.release();
	vertexindex_to_voxel.release();
	normals_buffer.release();

		
	//Sleep(2000);
	clock_t now;
	now = clock();


	printf("Elased time %f milli seconds\n", (double)((now - begin)/*/CLOCKS_PER_SEC*/));

	FILE* fp_qi;
	fp_qi = fopen("qi_volume.txt", "wt");

	fprintf(fp_qi, "qi tsdf w vx vy vz\n");


	const char *ext_ = strrchr(file_output, '.');
	if (strcmp(ext_, ".ply") == 0) {
		// write to ply file
		nw_cui_message_s("cu_voxel_rlc_to_mesh - write to a ply file...");
		FILE *fp = fopen(file_output, "wb");
		fprintf(fp, "ply\n");
		fprintf(fp, "format binary_little_endian 1.0\n");
		fprintf(fp, "element vertex %d\n", triangles_device.size());
		fprintf(fp, "property float32 x\n");
		fprintf(fp, "property float32 y\n");
		fprintf(fp, "property float32 z\n");
		fprintf(fp, "property float32 nx\n");
		fprintf(fp, "property float32 ny\n");
		fprintf(fp, "property float32 nz\n");
		fprintf(fp, "element face %d\n", triangles_device.size() / 3);
		fprintf(fp, "property list uint8 int32 vertex_indices\n");
		fprintf(fp, "end_header\n");
		for (int i = 0; i<triangles_device.size(); i++) {
			fwrite(&(h_vertices + i)->ver_pos, sizeof(float3), 1, fp);
			fwrite(h_normals + i, sizeof(float3), 1, fp);
		}
		uchar	prop_ = 3;
		int		face_[3];
		for (int i = 0; i<triangles_device.size() / 3; i++) {
			face_[0] = i * 3; face_[1] = face_[0] + 1; face_[2] = face_[1] + 1;
			fwrite(&prop_, 1, 1, fp);
			fwrite(face_, sizeof(int), 3, fp);
		}
		fclose(fp);
		nw_cui_message_e("done");
	}
	else if (strcmp(ext_, ".obj") == 0) {
		// write to obj file
		nw_cui_message_s("cu_voxel_rlc_to_mesh - write to an obj file...");
		FILE *fp = fopen(file_output, "w");
		/*for (int i = 0; i<triangles_device.size(); i++) {
			fprintf(fp, "v %f %f %f\n", h_vertices[i].x, h_vertices[i].y, h_vertices[i].z);
			fprintf(fp, "vn %f %f %f\n", h_normals[i].x, h_normals[i].y, h_normals[i].z);
		}
		for (int i = 0; i<triangles_device.size() / 3; i++) {
			fprintf(fp, "f %d %d %d\n", i * 3 + 1, i * 3 + 2, i * 3 + 3);
		}*/

		std::map<std::string, int> vertices_without_duplication;
		std::map<std::string, bool> bool_duplication;
	int size = triangles_device.size();
	std::string *vertices = new std::string[size];
	int count = 0;
		
		for(int i = 0; i<triangles_device.size(); i++) {

			char str[25];
			sprintf(str, "%.4f_%.4f_%.4f", h_vertices[i].ver_pos.x, h_vertices[i].ver_pos.y, h_vertices[i].ver_pos.z);

			std::string s(str);
			//std::map<std::string, int>::iterator dupl_it;  
			//dupl_it = vertices_without_duplication.find(s);
			//if (dupl_it == vertices_without_duplication.end())
			if (bool_duplication[s]!=TRUE)
			{

				fprintf(fp, "v %f %f %f\n", h_vertices[i].ver_pos.x, h_vertices[i].ver_pos.y, h_vertices[i].ver_pos.z);
				fprintf(fp, "vn %f %f %f\n", h_normals[i].x, h_normals[i].y, h_normals[i].z);

				vertices_without_duplication[s] = ++count;


				fprintf(fp_qi, "%d %hd %hd %d %d %d\n", count, h_vertices[i].tsdf, h_vertices[i].w, h_vertices[i].vox_pos.x, h_vertices[i].vox_pos.y, h_vertices[i].vox_pos.z);
			
				bool_duplication[s] = TRUE;
			}
			else
				printf("%d : v %f %f %f\n", i, h_vertices[i].ver_pos.x, h_vertices[i].ver_pos.y, h_vertices[i].ver_pos.z);
			vertices[i] = s;

		}
	

		fclose(fp_qi);

		for (int i = 0; i<triangles_device.size() / 3; i++) {
			int a, b, c;
			a = vertices_without_duplication[vertices[i * 3 + 0]]; b = vertices_without_duplication[vertices[i * 3 + 1]]; c = vertices_without_duplication[vertices[i * 3 + 2]];
			
			if (a == b || a == c || b == c) // 한 삼각형 안에 동일 버텍스 있는 삼각형 제외
			{
				printf("%d %d %d | ", a, b, c);
				std::cout << vertices[i * 3 + 0] << "/" << vertices[i * 3 + 1] << "/" << vertices[i * 3 + 2] << std::endl;
			}
			else
			{
				fprintf(fp, "f %d %d %d\n", vertices_without_duplication[vertices[i * 3 + 0]], vertices_without_duplication[vertices[i * 3 + 1]], vertices_without_duplication[vertices[i * 3 + 2]]);
			}
		
		}
		delete[] vertices;
		fclose(fp);
		nw_cui_message_e("done");
	}
	else {
		nw_cui_error("cu_voxel_rlc_to_mesh - Invalid file extension (%s)", ext_);
	}

	triangles_device.release();
	if (h_vertices != NULL)
	{
		free(h_vertices);
	}
	if (h_normals != NULL)
	{
		free(h_normals);
	}
}

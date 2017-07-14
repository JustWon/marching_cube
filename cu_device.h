/**
	\file		cu_device.h
	\brief		Device functions
	\author		Seongoh Lee
	\version	1.0
	\date		2011.12.09
*/

#ifndef _CU_DEVICE_H
#define _CU_DEVICE_H

//#include <cuda_runtime.h>
//#include <cutil_inline.h>

// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <helper_timer.h>

#include <cuda.h>
#include <npp.h>

#define cuda_function_header	__host__ __device__ __forceinline__

const int CU_BLOCK_MAX = 256;
const int CU_BLOCK_X = 16;
const int CU_BLOCK_Y = 16;

// types
#ifndef uchar
typedef unsigned char	uchar;
#endif
#ifndef ushort
typedef unsigned short	ushort;
#endif
#ifndef uint
typedef unsigned int	uint;
#endif

template<class T> inline void cu_malloc( T **d_ptr, size_t size )
{
	checkCudaErrors(cudaMalloc( d_ptr, size * sizeof(T) ));
}

void inline cu_free( void *d_ptr )
{
	checkCudaErrors(cudaFree( d_ptr ));
}

template<class T> inline void cu_malloc_host( T **h_ptr, size_t size, unsigned int flags = 0U )
{
	checkCudaErrors( cudaMallocHost( h_ptr, size * sizeof(T), flags ) );
}

void inline cu_free_host( void *h_ptr )
{
	checkCudaErrors(cudaFreeHost( h_ptr ));
}

template<class T> inline void cu_memcpy_device_to_host( T *src, T *dst, size_t size )
{
	checkCudaErrors(cudaMemcpy( (void*)dst, (const void*)src, size * sizeof(T), cudaMemcpyDeviceToHost ));
}

template<class T> inline void cu_memcpy_host_to_device( T *src, T *dst, size_t size )
{
	checkCudaErrors(cudaMemcpy( (void*)dst, (const void*)src, size * sizeof(T), cudaMemcpyHostToDevice ));
}

template<class T> inline void cu_memcpy_device_to_device( T *src, T *dst, size_t size )
{
	checkCudaErrors(cudaMemcpy( (void*)dst, (const void*)src, size * sizeof(T), cudaMemcpyDeviceToDevice ));
}

template<class T> inline void cu_memcpy_async_device_to_host( T *src, T *dst, size_t size )
{
	checkCudaErrors(cudaMemcpyAsync( (void*)dst, (const void*)src, size * sizeof(T), cudaMemcpyDeviceToHost ));
}

template<class T> inline void cu_memcpy_async_host_to_device( T *src, T *dst, size_t size )
{
	checkCudaErrors(cudaMemcpyAsync( (void*)dst, (const void*)src, size * sizeof(T), cudaMemcpyHostToDevice ));
}

template<class T> inline void cu_memcpy_async_device_to_device( T *src, T *dst, size_t size )
{
	checkCudaErrors(cudaMemcpyAsync( (void*)dst, (const void*)src, size * sizeof(T), cudaMemcpyDeviceToDevice ));
}


// Note that pitch, spitch, dpitch (in bytes)
template<class T> void inline cu_malloc_2d( T **d_ptr, size_t *pitch, size_t width, size_t height )
{
	checkCudaErrors( cudaMallocPitch( d_ptr, pitch, width * sizeof(T), height) );        
}

template<class T> void inline cu_memcpy_2d_device_to_host( T *src, size_t spitch, T *dst, size_t dpitch, size_t width, size_t height )
{
	checkCudaErrors( cudaMemcpy2D( (void*)dst, dpitch, (const void*)src, spitch, width * sizeof(T), height, cudaMemcpyDeviceToHost) );        
}

template<class T> void inline cu_memcpy_2d_host_to_device( T *src, size_t spitch, T *dst, size_t dpitch, size_t width, size_t height )
{
	checkCudaErrors( cudaMemcpy2D( (void*)dst, dpitch, (const void*)src, spitch, width * sizeof(T), height, cudaMemcpyHostToDevice) );        
}

template<class T> void inline cu_memcpy_2d_device_to_device( T *src, size_t spitch, T *dst, size_t dpitch, size_t width, size_t height )
{
	checkCudaErrors( cudaMemcpy2D( (void*)dst, dpitch, (const void*)src, spitch, width * sizeof(T), height, cudaMemcpyDeviceToDevice) );        
}

void inline cu_error()
{
	checkCudaErrors( cudaGetLastError() );
}

void inline cu_sync()
{
	checkCudaErrors( cudaDeviceSynchronize() );
}

inline int cu_initialize()
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		printf( "CUDA error: no devices supporting CUDA.\n" );
		exit(-1);
	}
	int dev = gpuGetMaxGflopsDeviceId();
	cudaDeviceProp cu_prop;
	cudaGetDeviceProperties(&cu_prop, dev);
	printf( "device name:               %s\n", cu_prop.name );
	printf( "total global memory:       %u\n", cu_prop.totalGlobalMem );
	printf( "shared memory per block:   %u\n", cu_prop.sharedMemPerBlock );
	printf( "warp size:                 %d\n", cu_prop.warpSize );
	printf( "max threads per block:     %d\n", cu_prop.maxThreadsPerBlock );
	printf( "max threads dim:           %d, %d, %d\n", cu_prop.maxThreadsDim[0], cu_prop.maxThreadsDim[1], cu_prop.maxThreadsDim[2] );
	printf( "max grid size:             %d, %d, %d\n", cu_prop.maxGridSize[0], cu_prop.maxGridSize[1], cu_prop.maxGridSize[2] );

	checkCudaErrors(cudaSetDevice(dev));
	return dev;
}

inline void cu_terminate()
{
	checkCudaErrors( cudaDeviceReset() );
}

#endif /* !_CU_DEVICE_H */

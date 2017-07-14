/**
\file		cu_marching_cubes_v1.cu
\brief		Cuda marching cubes library (modified version of the PCL) source

\author		Seong-Oh Lee
\version	1.0
\date		2012.04.10
*/

#include "device_launch_parameters.h"
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <helper_math.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "cu_marching_cubes_v1.h"

////////
// CU //
////////

#include "pcl/gpu/utils/device/block.hpp"
#include "pcl/gpu/utils/device/warp.hpp"

using namespace pcl::device;

int3	volume_size;

//texture<int, 1, cudaReadModeElementType> edgeTex;
texture<int, 1, cudaReadModeElementType> triTex;
texture<int, 1, cudaReadModeElementType> numVertsTex;

void bindTextures (const int */*edgeBuf*/, const int *triBuf, const int *numVertsBuf)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	//checkCudaErrors(cudaBindTexture(0, edgeTex, edgeBuf, desc) );
	checkCudaErrors (cudaBindTexture (0, triTex, triBuf, desc) );
	checkCudaErrors (cudaBindTexture (0, numVertsTex, numVertsBuf, desc) );
}
void unbindTextures ()
{
	//checkCudaErrors( cudaUnbindTexture(edgeTex) );
	checkCudaErrors ( cudaUnbindTexture (numVertsTex) );
	checkCudaErrors ( cudaUnbindTexture (triTex) );
}

__device__ int global_count = 0;
__device__ int output_count;
__device__ unsigned int blocks_done = 0;

struct CubeIndexEstimator
{
	ptr_step<cu_voxel_t> volume;

	static __device__ __forceinline__ float isoValue() { return 0.f; }

	__device__ __forceinline__ void
		readTsdf (int VOLUME_Y, int x, int y, int z, float& tsdf, int& weight) const
	{
		const cu_voxel_t *ptr = &volume.ptr(VOLUME_Y * z + y)[x];
		tsdf = d_unpack_tsdf( ptr->tsdf );
		
		//weight = d_unpack_weight( ptr->w );
		weight = (int)ptr->w;
		
	}

	__device__ __forceinline__ int
		computeCubeIndex (int VOLUME_Y, int x, int y, int z, float f[8]) const
	{
		int weight;
		readTsdf (VOLUME_Y, x,     y,     z,     f[0], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x + 1, y,     z,     f[1], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x + 1, y + 1, z,     f[2], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x,     y + 1, z,     f[3], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x,     y,     z + 1, f[4], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x + 1, y,     z + 1, f[5], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x + 1, y + 1, z + 1, f[6], weight); if (weight == 0) return 0;
		readTsdf (VOLUME_Y, x,     y + 1, z + 1, f[7], weight); if (weight == 0) return 0;
		// calculate flag indicating if each vertex is inside or outside isosurface
		int cubeindex;
		cubeindex = int(f[0] < isoValue());
		cubeindex += int(f[1] < isoValue()) * 2;
		cubeindex += int(f[2] < isoValue()) * 4;
		cubeindex += int(f[3] < isoValue()) * 8;
		cubeindex += int(f[4] < isoValue()) * 16;
		cubeindex += int(f[5] < isoValue()) * 32;
		cubeindex += int(f[6] < isoValue()) * 64;
		cubeindex += int(f[7] < isoValue()) * 128;

		return cubeindex;
	}
};

struct OccupiedVoxels : public CubeIndexEstimator
{
	enum
	{
		CTA_SIZE_X = 32,
		CTA_SIZE_Y = 8,
		CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

		WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE,
	};

	mutable int* voxels_indeces;
	mutable int* vetexes_number;
	int max_size;

	__device__ __forceinline__ void
		operator () ( int3 volume_size ) const
	{
		int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
		if (__all (x >= volume_size.x) || __all (y >= volume_size.y))
			return;

		int ftid = Block::flattenedThreadId ();
		int warp_id = Warp::id();
		int lane_id = Warp::laneId();

		volatile __shared__ int warps_buffer[WARPS_COUNT];

		for (int z = 0; z < volume_size.z - 1; z++)
		{
			int numVerts = 0;;
			if (x + 1 < volume_size.x && y + 1 < volume_size.y)
			{
				float field[8];
				int weight;
				
				int cubeindex = computeCubeIndex (volume_size.y, x, y, z, field);
				// read number of vertices from texture
				numVerts = (cubeindex == 0 || cubeindex == 255) ? 0 : tex1Dfetch (numVertsTex, cubeindex);
			}

			int total = __popc (__ballot (numVerts > 0));
			if (total == 0)
				continue;

			if (lane_id == 0)
			{
				int old = atomicAdd (&global_count, total);
				warps_buffer[warp_id] = old;
			}
			int old_global_voxels_count = warps_buffer[warp_id];

			int offs = Warp::binaryExclScan (__ballot (numVerts > 0));

			if (old_global_voxels_count + offs < max_size && numVerts > 0)
			{
				voxels_indeces[old_global_voxels_count + offs] = volume_size.y * volume_size.x * z + volume_size.x * y + x;
				vetexes_number[old_global_voxels_count + offs] = numVerts;
			}

			bool full = old_global_voxels_count + total >= max_size;

			if (full)
				break;

		} /* for(int z = 0; z < VOLUME_Z - 1; z++) */


		/////////////////////////
		// prepare for future scans
		if (ftid == 0)
		{
			unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
			unsigned int value = atomicInc (&blocks_done, total_blocks);

			//last block
			if (value == total_blocks - 1)
			{
				output_count = min (max_size, global_count);
				blocks_done = 0;
				global_count = 0;
			}
		} 
	} /* operator () */
};
__global__ void getOccupiedVoxelsKernel (const OccupiedVoxels ov, int3 volume_size ) { ov ( volume_size ); }

int getOccupiedVoxels (const ptr_step<cu_voxel_t>& volume, cu_array_2d<int>& occupied_voxels)
{
	OccupiedVoxels ov;
	ov.volume = volume;
	ov.voxels_indeces = occupied_voxels.ptr (0);
	ov.vetexes_number = occupied_voxels.ptr (1);
	ov.max_size = occupied_voxels.cols ();

	dim3 block (OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y);
	dim3 grid (int_div_up (volume_size.x, block.x), int_div_up (volume_size.y, block.y));

	//cudaFuncSetCacheConfig(getOccupiedVoxelsKernel, cudaFuncCachePreferL1);
	//printFuncAttrib(getOccupiedVoxelsKernel);

	getOccupiedVoxelsKernel<<<grid, block>>>(ov, volume_size );
	cu_error();
	cu_sync();

	int size;
	checkCudaErrors ( cudaMemcpyFromSymbol (&size, output_count, sizeof(size)) );
	return size;
}

int computeOffsetsAndTotalVertexes (cu_array_2d<int>& occupied_voxels)
{
	thrust::device_ptr<int> beg (occupied_voxels.ptr (1));
	thrust::device_ptr<int> end = beg + occupied_voxels.cols ();

	thrust::device_ptr<int> out (occupied_voxels.ptr (2));
	thrust::exclusive_scan (beg, end, out);

	int lastElement, lastScanElement;

	cu_array<int> last_elem (occupied_voxels.ptr(1) + occupied_voxels.cols () - 1, 1);
	cu_array<int> last_scan (occupied_voxels.ptr(2) + occupied_voxels.cols () - 1, 1);

	last_elem.download (&lastElement);
	last_scan.download (&lastScanElement);

	return lastElement + lastScanElement;
}


struct TrianglesGenerator : public CubeIndexEstimator
{
	enum { CTA_SIZE = 256 };

	const int* occupied_voxels;
	const int* vertex_ofssets;
	int voxels_count;
	float3 cell_size;
	
	mutable vertex_voxel *output;

	mutable int3* vetexindex_to_voxel; //inyeop
	mutable short* ver_voxel;//inyeop
	mutable int3 vol_size;//inyeop

	__device__ __forceinline__ float3
		getNodeCoo (int x, int y, int z) const
	{
		float3 coo = make_float3 (x, y, z);
		coo = float3_add_c( coo, 0.5f ); //shift to volume cell center;

		coo.x *= cell_size.x;
		coo.y *= cell_size.y;
		coo.z *= cell_size.z;

		return coo;
	}

	__device__ __forceinline__ float3
		vertex_interp (float3 p0, float3 p1, float f0, float f1) const
	{        
		float t = (isoValue() - f0) / (f1 - f0 + 1e-15f);
		float x = p0.x + t * (p1.x - p0.x);
		float y = p0.y + t * (p1.y - p0.y);
		float z = p0.z + t * (p1.z - p0.z);
		return make_float3 (x, y, z);
	}

	__device__ __forceinline__ void
		operator () ( int3 volume_size ) const
	{
		int tid = threadIdx.x;
		int idx = blockIdx.x * CTA_SIZE + tid;

		if (idx >= voxels_count)
			return;

		int voxel = occupied_voxels[idx];

		int z = voxel / (volume_size.x * volume_size.y);
		int y = (voxel - z * volume_size.x * volume_size.y) / volume_size.x;
		int x = (voxel - z * volume_size.x * volume_size.y) - y * volume_size.x;

		//inyeop
		vol_size.x = volume_size.x; vol_size.y = volume_size.y; vol_size.z = volume_size.z;

		float f[8];
		int cubeindex = computeCubeIndex (volume_size.y, x, y, z, f);

	

		// calculate cell vertex positions
		float3 v[8];
		v[0] = getNodeCoo (x, y, z);
		v[1] = getNodeCoo (x + 1, y, z); 
		v[2] = getNodeCoo (x + 1, y + 1, z);
		v[3] = getNodeCoo (x, y + 1, z);
		v[4] = getNodeCoo (x, y, z + 1);
		v[5] = getNodeCoo (x + 1, y, z + 1);
		v[6] = getNodeCoo (x + 1, y + 1, z + 1);
		v[7] = getNodeCoo (x, y + 1, z + 1);

		// find the vertices where the surface intersects the cube
		// use shared memory to avoid using local
		__shared__ float3 vertlist[12][CTA_SIZE];

		vertlist[0][tid] = vertex_interp (v[0], v[1], f[0], f[1]);
		vertlist[1][tid] = vertex_interp (v[1], v[2], f[1], f[2]);
		vertlist[2][tid] = vertex_interp (v[2], v[3], f[2], f[3]);
		vertlist[3][tid] = vertex_interp (v[3], v[0], f[3], f[0]);
		vertlist[4][tid] = vertex_interp (v[4], v[5], f[4], f[5]);
		vertlist[5][tid] = vertex_interp (v[5], v[6], f[5], f[6]);
		vertlist[6][tid] = vertex_interp (v[6], v[7], f[6], f[7]);
		vertlist[7][tid] = vertex_interp (v[7], v[4], f[7], f[4]);
		vertlist[8][tid] = vertex_interp (v[0], v[4], f[0], f[4]);
		vertlist[9][tid] = vertex_interp (v[1], v[5], f[1], f[5]);
		vertlist[10][tid] = vertex_interp (v[2], v[6], f[2], f[6]);
		vertlist[11][tid] = vertex_interp (v[3], v[7], f[3], f[7]);
		__syncthreads();

		// output triangle vertices
		int numVerts = tex1Dfetch (numVertsTex, cubeindex);

		
		for (int i = 0; i < numVerts; i += 3)
		{
			int index = vertex_ofssets[idx] + i;

			int v1 = tex1Dfetch (triTex, (cubeindex * 16) + i + 0);
			int v2 = tex1Dfetch (triTex, (cubeindex * 16) + i + 1);
			int v3 = tex1Dfetch (triTex, (cubeindex * 16) + i + 2);

			store_point (output, index + 2, vertlist[v1][tid]);
			store_point (output, index + 1, vertlist[v2][tid]);
			store_point (output, index + 0, vertlist[v3][tid]);
		}
	}

	__device__ __forceinline__ void
		store_point (vertex_voxel *ptr, int index, const float3& point) const {
			ptr[index].ver_pos = make_float3 (point.x, point.y, point.z);
			
					
			/****************************inyeop**********************************************/
			float3 coo = make_float3(point.x, point.y, point.z);
			
			coo.x /= cell_size.x;
			coo.y /= cell_size.y;
			coo.z /= cell_size.z;
			
			coo = float3_sub_c(coo, 0.5f); //shift to volume cell center;
	int x, y, z;
			x = int(floor(coo.x)); y = int(floor(coo.y)); z = int(floor(coo.z));
			vetexindex_to_voxel[index] = make_int3(floor(coo.x), floor(coo.y), floor(coo.z));
			/****************************inyeop**********************************************/

			int i = z*vol_size.y*vol_size.x + y*vol_size.x + x;                                                                                     
			//int i = z * 256 * 256 + y * 256 + x;
			//ver_voxel[i]= -1;
			 const cu_voxel_t *vptr = &volume.ptr(vol_size.y * z + y)[x];
			 ver_voxel[i] = vptr->tsdf;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
		 int weight;
//			 readTsdf(vol_size.y, x, y, z, ver_voxel[i], weight);
			

		 ptr[index].vox_pos.x = x; ptr[index].vox_pos.y = y; ptr[index].vox_pos.z = z;
		 ptr[index].tsdf = vptr->tsdf;
		 ptr[index].w = vptr->w;
		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
	}
};
__global__ void
	trianglesGeneratorKernel (const TrianglesGenerator tg, int3 volume_size) {tg ( volume_size ); }

void generateTriangles(const ptr_step<cu_voxel_t>& volume, cu_array_2d<int>& occupied_voxels, const float cell_size, cu_array<vertex_voxel>& output, cu_array<int3>& vertexindex_to_voxel, cu_array<short>& ver_voxel/*inyeop*/)
{
	TrianglesGenerator tg;

	tg.volume = volume;
	tg.occupied_voxels = occupied_voxels.ptr (0);
	tg.vertex_ofssets = occupied_voxels.ptr (2);
	tg.voxels_count = occupied_voxels.cols ();
	tg.cell_size.x = cell_size;
	tg.cell_size.y = cell_size;
	tg.cell_size.z = cell_size;
	tg.output = output;
	tg.vetexindex_to_voxel = vertexindex_to_voxel;
	tg.ver_voxel = ver_voxel;


	dim3 block (TrianglesGenerator::CTA_SIZE);
	dim3 grid (int_div_up (tg.voxels_count, block.x));

	trianglesGeneratorKernel<<<grid, block>>>(tg, volume_size);
	cu_error();	cu_sync();



}

/////////
// CPP //
/////////

extern const int edgeTable[256];
extern const int triTable[256][16]; 
extern const int numVertsTable[256];

MarchingCubes::MarchingCubes()
{
	edgeTable_.upload(edgeTable, 256);
	numVertsTable_.upload(numVertsTable, 256);
	triTable_.upload(&triTable[0][0], 256 * 16);    
}

MarchingCubes::~MarchingCubes() {}

//cu_array<float3> 
cu_array<vertex_voxel>
MarchingCubes::run(cu_voxel& tsdf, cu_array<vertex_voxel>& triangles_buffer, cu_array<int3>& vertexindex_to_voxel, cu_array<short>& ver_voxel/*inyeop*/)
{
	
	volume_size = tsdf.size;
	cu_array_2d<cu_voxel_t> volume_( tsdf.size.y * tsdf.size.z, tsdf.size.x, tsdf.data, tsdf.size.x * sizeof(cu_voxel_t) );

	if (triangles_buffer.empty())
		triangles_buffer.create(DEFAULT_TRIANGLES_BUFFER_SIZE);
	
	/*****************inyeop***************************************/
	if (vertexindex_to_voxel.empty())
		vertexindex_to_voxel.create(DEFAULT_TRIANGLES_BUFFER_SIZE);
	/*****************inyeop***************************************/
	
	
	printf("\n volume_size : %d %d %d\n", volume_size.x, volume_size.y, volume_size.z);
	/*****************inyeop***************************************/
	if (ver_voxel.empty())
		ver_voxel.create(volume_size.x*volume_size.y*volume_size.z); // <----
	
	/*****************inyeop***************************************/
	cudaMemset(ver_voxel.ptr(), 0, volume_size.x*volume_size.y*volume_size.z*sizeof(short));
	
	/*	short* h_ver_voxel = alloc_x(short, ver_voxel.size());
	ver_voxel.download(h_ver_voxel);
	*/
	/*	short* h_ver_voxel;
	h_ver_voxel = (short*)malloc(volume_size.x*volume_size.y*volume_size.z*sizeof(short));
	cudaMemcpy(h_ver_voxel, ver_voxel.ptr(), volume_size.x*volume_size.y*volume_size.z*sizeof(short), cudaMemcpyDeviceToHost);
	*/

	occupied_voxels_buffer_.create(3, triangles_buffer.size() / 3);    

	bindTextures(edgeTable_, triTable_, numVertsTable_);

	int active_voxels = getOccupiedVoxels(volume_, occupied_voxels_buffer_);  
	if(!active_voxels) {
		unbindTextures();
		return cu_array<vertex_voxel>();
	}

	cu_array_2d<int> occupied_voxels(3, active_voxels, occupied_voxels_buffer_.ptr(), occupied_voxels_buffer_.step());

	int total_vertexes = computeOffsetsAndTotalVertexes(occupied_voxels);
	
	generateTriangles(volume_, occupied_voxels, tsdf.grid_s, (cu_array<vertex_voxel>&)triangles_buffer, (cu_array<int3>&)vertexindex_to_voxel, (cu_array<short>&)ver_voxel/*ineyop*/);


	unbindTextures();
	return cu_array<vertex_voxel>(triangles_buffer.ptr(), total_vertexes);
}

// edge table maps 8-bit flag representing which cube vertices are inside
// the isosurface to 12-bit number indicating which edges are intersected
const int edgeTable[256] = 
{
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

// triangle table maps same cube vertex index to a list of up to 5 triangles
// which are built from the interpolated edge vertices
const int triTable[256][16] = 
{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// number of vertices for each case above
const int numVertsTable[256] = 
{
	0,
	3,
	3,
	6,
	3,
	6,
	6,
	9,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	6,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	6,
	9,
	9,
	6,
	9,
	12,
	12,
	9,
	9,
	12,
	12,
	9,
	12,
	15,
	15,
	6,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	12,
	12,
	15,
	12,
	15,
	15,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	6,
	9,
	9,
	12,
	12,
	15,
	12,
	15,
	9,
	6,
	9,
	12,
	12,
	9,
	12,
	15,
	9,
	6,
	12,
	15,
	15,
	12,
	15,
	6,
	12,
	3,
	3,
	6,
	6,
	9,
	6,
	9,
	9,
	12,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	9,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	6,
	12,
	9,
	12,
	9,
	15,
	6,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	12,
	12,
	15,
	12,
	15,
	15,
	12,
	9,
	12,
	12,
	9,
	12,
	15,
	15,
	12,
	12,
	9,
	15,
	6,
	15,
	12,
	6,
	3,
	6,
	9,
	9,
	12,
	9,
	12,
	12,
	15,
	9,
	12,
	12,
	15,
	6,
	9,
	9,
	6,
	9,
	12,
	12,
	15,
	12,
	15,
	15,
	6,
	12,
	9,
	15,
	12,
	9,
	6,
	12,
	3,
	9,
	12,
	12,
	15,
	12,
	15,
	9,
	12,
	12,
	15,
	15,
	6,
	9,
	12,
	6,
	3,
	6,
	9,
	9,
	6,
	9,
	12,
	6,
	3,
	9,
	6,
	12,
	3,
	6,
	3,
	3,
	0,
};

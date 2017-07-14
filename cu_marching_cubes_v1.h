/**
\file		cu_marching_cubes_v1.h
\brief		Cuda marching cubes library (modified version of the PCL) header

\author		Seong-Oh Lee
\version	1.0
\date		2012.04.10
*/

#ifndef _CU_MARCHING_CUBES_V1_H
#define _CU_MARCHING_CUBES_V1_H

#include "cu_memory.hpp"
#include "cu_kinect_fusion_v1.h"

#include "structure.h"
#include "cu_marching_cubes_v1.h"
#include "helper_math.h"
#include "con_ui.h"
//#include "cu_memory.hpp"
#include "cu_device.h"
#include "cu_kinect_fusion_v1.h"
#include "cu_vector_functions.h"

/** \brief MarchingCubes implements MarchingCubes functionality for TSDF volume on GPU
* \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/
class MarchingCubes
{
public:
	/** \brief Default size for triangles buffer */
	enum
	{ 
		POINTS_PER_TRIANGLE = 3,
		DEFAULT_TRIANGLES_BUFFER_SIZE = 9 * 1000 * 1000 * POINTS_PER_TRIANGLE      
	};

	/** \brief Default constructor */
	MarchingCubes();

	/** \brief Destructor */
	~MarchingCubes();

	/** \brief Runs marching cubes triangulation.
	* \param[in] kinfu KinFu tracker class to take tsdf volume from
	* \param[in] triangles_buffer Buffer for triangles. Its size determines max extracted triangles. If empty, it will be allocated with default size will be used.          
	* \return Array with triangles. Each 3 consequent poits belond to a single triangle. The returned array points to 'triangles_buffer' data.
	*/
//	cu_array<float3> 
	cu_array<vertex_voxel>
		run(cu_voxel& tsdf, cu_array<vertex_voxel>& triangles_buffer, cu_array<int3>& vertexindex_to_voxel, cu_array<short>& ver_voxel/*inyeop*/);

private:
	/** \brief Edge table for marching cubes  */
	cu_array<int> edgeTable_;

	/** \brief Number of vertextes table for marching cubes  */
	cu_array<int> numVertsTable_;

	/** \brief Triangles table for marching cubes  */
	cu_array<int> triTable_;     

	/** \brief Temporary buffer used by marching cubes (first row stores occuped voxes id, second number of vetexes, third poits offsets */
	cu_array_2d<int> occupied_voxels_buffer_;
};

#endif // _CU_MARCHING_CUBES_V1_H

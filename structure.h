#ifndef _STRUCTURE_H//inyeop
#define _STRUCTURE_H

#include "cu_math_functions.h"


typedef struct _CU_COLOR_T
{
	uchar	p[3];
} cu_color_t;



typedef struct _CU_VOXEL_T {
	short	tsdf;	// normalized tsdf
	ushort	w;		// weight
	//uchar3	color;	// color (BGR)
} cu_voxel_t;

typedef struct _CU_VOXEL {
	int3		size;
	int			size_xy;	// size.x * size.y
	float		min_t;		// min truncation e.g. 0.06 m
	float		max_t;		// max truncation e.g. 0.06 m
	float		grid_s;		// grid size e.g. 0.005 m
	cu_voxel_t	*data;		// volume data (device) pointer, [z][y][x]
} cu_voxel;

typedef struct _VERTEX_VOXEL
{
	float3 ver_pos;
	int3 vox_pos;
	short tsdf;
	short w;
}vertex_voxel;

#define cu_voxel_access_z(v,_z) ((v)->data + (_z) * (v)->size_xy) /// [z][][]
#define cu_voxel_access_yz(v,_y,_z) (cu_voxel_access_z(v,_z) + (_y) * (v)->size.x) ///< [z][y][]
#define cu_voxel_access_xyz(v,_x,_y,_z) (cu_voxel_access_yz(v,_y,_z) + (_x)) ///< [z][y][x]


#endif
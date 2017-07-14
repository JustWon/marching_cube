#include <iostream>
#include "cu_kinect_fusion_v1.h"
#include "cu_marching_cubes_v1.h"

int main()
{
	//cu_voxel_rlc_to_mesh("512dim.vol","512dim.ply", 512);
	cu_voxel_rlc_to_mesh("1024dim.vol", "1024dim.ply", 1024);

	return 0;
}
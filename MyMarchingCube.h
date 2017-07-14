#pragma once


class MyMarchingCube
{
public:
	MyMarchingCube(void* _volume_data, unsigned int _size_x, unsigned int _size_y, unsigned int _size_z, float grid_s, const char* outfilepath);
	MyMarchingCube(char *sdffilepath="sdf.txt", char* outfilepath="volume_marching_cube.obj" );
	~MyMarchingCube(); 
};


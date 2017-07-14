/**
	\file		cu_geometry_functions.h
	\brief		geometry functions
	\author		Seongoh Lee
	\version	1.0
	\date		2011.12.09
*/

#ifndef _CU_GEOMETRY_FUNCTIONS_H
#define _CU_GEOMETRY_FUNCTIONS_H

#include "cu_vector_functions.h"

/////////////
// Line 2D //
/////////////

/// \param d	[in] directional vector
/// \param p	[in] a point
/// \return line [a,b,c]^t
cuda_function_header float3 float_vector_to_line_2d( float2 d, float2 p )
{
	return make_float3( d.y, -d.x, -d.y * p.x + d.x * p.y );
}

//////////////
// Triangle //
//////////////

#define vector_triangle_area2_2d(a,b,c)	(((a).x-(c).x)*((b).y-(a).y)-((a).x-(b).x)*((c).y-(a).y))
#define vector_triangle_area_2d(a,b,c)	(vector_triangle_area2_2d(a,b,c)/2);

#define vector_barycentric_inside(a) ((a).x>0 && (a).x<1 && (a).y>0 && (a).y<1 && (a).z>0 && (a).z<1) ///< TRUE if a point lies inside the triangle.
#define vector_barycentric_outside(a) ((a).x<0 || (a).x>1 || (a).y<0 || (a).y>1 || (a).z<0 || (a).z>1) ///< TRUE if a point lies outside the triangle.

/// Compute barycentric coordinates (x, y, z) for point p with respect to triangle (a, b, c)
/// Note that if p exist inside of the triangle, x, y, z are in [0,1]
inline cuda_function_header float3 float_barycentric_from_2d( float2 point, float2 triangle_a, float2 triangle_b, float2 triangle_c, float area2 )
{
	float3	barycentric_p;
	if( area2 == 0 ) area2 = vector_triangle_area2_2d( triangle_a, triangle_b, triangle_c ); // constant at each triangle
	barycentric_p.x = vector_triangle_area2_2d( point, triangle_b, triangle_c ) / area2;
	barycentric_p.y = vector_triangle_area2_2d( point, triangle_c, triangle_a ) / area2;
	//barycentric_p.z = vector_triangle_area2_2d( point, triangle_a, triangle_b ) / area2;
	barycentric_p.z = ( 1.0f - barycentric_p.x - barycentric_p.y );
	return barycentric_p;
}

/// Compute point p with respect to triangle (a, b, c) for barycentric coordinates (u, v, w)
inline cuda_function_header float3 float_barycentric_interpolation_3d( float3 barycentric_p, float3 triangle_a, float3 triangle_b, float3 triangle_c )
{
	float3	point;
	point.x = ( barycentric_p.x * triangle_a.x + barycentric_p.y * triangle_b.x + barycentric_p.z * triangle_c.x );
	point.y = ( barycentric_p.x * triangle_a.y + barycentric_p.y * triangle_b.y + barycentric_p.z * triangle_c.y );
	point.z = ( barycentric_p.x * triangle_a.z + barycentric_p.y * triangle_b.z + barycentric_p.z * triangle_c.z );
	return point;
}

///////////
// Plane //
///////////

/// Planar Fitting using Three 3D Points (plane (p): p.x * x + p.y * y + p.z * z + p.w = 0)

/// Note that 3 points, 1 VP and 2 points, 2 VPs and 1 point
inline cuda_function_header float4 float_plane_fit_3pt( float3 p1, float3 p2, float3 p3 )
{
	float4	plane;
	float3	p13 = float3_sub( p1, p3 );
	float3	p23 = float3_sub( p2, p3 );
	vector3_cross(plane, p13, p23);
	vector3_cross(p13, p1, p2);
	plane.w = -float3_dot( p3, p13 );
	return plane;
}

/// Distance from a point to a plane (+/-)
inline cuda_function_header float float_plane_point_distance( float4 p, float3 x )
{
	float d = vector3_dot(p, x) + p.w;
	float n = float3_norm2(p);
	return ( d / n );
}

/// \param v_q	[in] a point on a plane
/// \param n_q	[in] n(q): a normalized normal vector on the q
/// \param v_p	[in] an input point
/// \note dot((v_p-v_q),n_q)
inline cuda_function_header float float_surface_point_to_point_distance( float3 v_q, float3 n_q, float3 v_p )
{
	return float3_dot( float3_sub( v_p, v_q ), n_q );
}

/// pinhole camera intrinsic parameters

typedef struct _float_camera_intrinsic
{
	float2	fc; ///< the focal lengths (aspect ratio = fc.y/fc.x)
	float2	cc; ///< a principal point
	float	dc[5]; ///< bouguet's model (radial and tangential: k1, k2, p1, p2, k3)
} float_camera_intrinsic;

/// Note that R^-1 = R^T.
typedef struct _float_camera_extrinsic
{
	float33	R; ///< a rotation matrix (column-major)
	float3	t; ///< a translation vector
} float_camera_extrinsic;

inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_default()
{
	float_camera_extrinsic	ce;
	matrix33_identity( ce.R.p );
	vector3_set( ce.t, 0.f );
	return ce;
}

inline cuda_function_header float3 float_camera_extrinsic_forward( float_camera_extrinsic ce, float3 p )
{
	float3	q = float33_mv( ce.R, p );
	q = float3_add( q, ce.t );
	return q;
}

inline cuda_function_header float3 float_camera_extrinsic_backward( float_camera_extrinsic ce, float3 p )
{
	p = float3_sub( p, ce.t );
	return float33_tv( ce.R, p );
}

inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_inverse( float_camera_extrinsic src )
{
	float_camera_extrinsic	dst;
	dst.R = float33_t( src.R );
	dst.t = float33_mv( dst.R, src.t );
	dst.t = float3_negation( dst.t );
	return dst;
}

/// Relative transformation between two rotation-and-shift transformations

/// T1=AC, T2=BC, T=AB
/// \param R	[out] the rotation matrix of the superposition
/// \param t	[out] the translation vector of the superposition
/// \param R1	[in] the 1st rotation matrix
/// \param t1	[in] the 1st translation vector
/// \param R2	[in] the 2nd rotation matrix
/// \param t2	[in] the 2nd translation vector
inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_relative( float_camera_extrinsic p_1, float_camera_extrinsic p_2 )
{
	float_camera_extrinsic	p_r;
	p_r.R = float33_tm( p_2.R, p_1.R );
	p_r.t = float3_sub( p_1.t, p_2.t );
	p_r.t = float33_tv( p_2.R, p_r.t );
	return p_r;
}

/// Combines two rotation-and-shift transformations

/// R2(R1 X + t1) + t2 = R2R1 X + R2 t1 + t2
/// \param R	[out] the rotation matrix of the superposition
/// \param t	[out] the translation vector of the superposition
/// \param R1	[in] the 1st rotation matrix
/// \param t1	[in] the 1st translation vector
/// \param R2	[in] the 2nd rotation matrix
/// \param t2	[in] the 2nd translation vector
inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_compose( float_camera_extrinsic p_1, float_camera_extrinsic p_2 )
{
	float_camera_extrinsic	p_c;
	p_c.R = float33_mm( p_2.R, p_1.R );
	p_c.t = float33_mv( p_2.R, p_1.t );
	p_c.t = float3_add( p_c.t, p_2.t );
	return p_c;
}

//inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_to_opengl( float_camera_extrinsic src )
//{
//	src.R.p[0].y *= -1.f; src.R.p[1].y *= -1.f; src.R.p[2].y *= -1.f;
//	src.R.p[0].z *= -1.f; src.R.p[1].z *= -1.f; src.R.p[2].z *= -1.f;
//	return src;
//}
//
//inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_from_opengl( float_camera_extrinsic src )
//{
//	float_camera_extrinsic	dst;
//	return dst;
//}

/// Three point registration: compute R and T
/// p2_i = R * p1_i + T, i=1,2,3
inline cuda_function_header float_camera_extrinsic float_camera_extrinsic_relative_from_three_point_pairs(
	float3 p1_1, float3 p1_2, float3 p1_3, float3 p2_1, float3 p2_2, float3 p2_3 )
{
	float_camera_extrinsic	p_r;
	float3	t1;
	float33	R1, R2;
	// R1
	R1.p[0] = float3_unit( float3_sub( p1_2, p1_1 ) ); t1 = float3_sub( p1_3, p1_1 );
	R1.p[1] = float3_unit( float3_sub( t1, float3_mul_c( R1.p[0], float3_dot( t1, R1.p[0] ) ) ) );
	R1.p[2] = float3_cross( R1.p[0], R1.p[1] );
	// R2
	R2.p[0] = float3_unit( float3_sub( p2_2, p2_1 ) ); t1 = float3_sub( p2_3, p2_1 );
	R2.p[1] = float3_unit( float3_sub( t1, float3_mul_c( R2.p[0], float3_dot( t1, R2.p[0] ) ) ) );
	R2.p[2] = float3_cross( R2.p[0], R2.p[1] );
	// R
	p_r.R = float33_mt( R2, R1 );
	// T
	p_r.t = float3_sub( p2_1, float33_mv( p_r.R, p1_1 ) );
	
	return p_r;
}

inline cuda_function_header float2 float_camera_intrinsic_set_cc( int2 size )
{
	return make_float2( ( size.x - 1 ) / 2.0f, ( size.y - 1 ) / 2.0f );
}

inline cuda_function_header float_camera_intrinsic float_camera_intrinsic_pyramid_down( float_camera_intrinsic src )
{
	float_camera_intrinsic	dst;
	dst.fc = float2_div_c( src.fc, 2.0f );
	dst.cc = float2_div_c( src.cc, 2.0f );
	vector1s_copy( dst.dc, src.dc, 5 );
	return dst;
}

inline cuda_function_header float2 float_camera_intrinsic_project_2( float_camera_intrinsic ci, float2 p )
{
	return float2_add( float2_mul( ci.fc, p ), ci.cc );
}

inline cuda_function_header float2 float_camera_intrinsic_project_3( float_camera_intrinsic ci, float3 p )
{
	return float_camera_intrinsic_project_2( ci, float2_make_inhomogeneous(p) );
}

inline cuda_function_header float2 float_camera_intrinsic_unproject_2( float_camera_intrinsic ci, float2 p )
{
	return float2_div( float2_sub( p, ci.cc ), ci.fc );
}

inline cuda_function_header float3 float_camera_intrinsic_unproject_3( float_camera_intrinsic ci, float2 p, float v )
{
	p = float_camera_intrinsic_unproject_2( ci, p );
	return float2_make_homogeneous( p, v );
}

inline cuda_function_header float33 float_camera_intrinsic_to_K( float_camera_intrinsic ci )
{
	float33	K;
	K.p[0] = make_float3( ci.fc.x, 0, 0 );
	K.p[1] = make_float3( 0, ci.fc.y, 0 );
	K.p[2] = make_float3( ci.cc.x, ci.cc.y, 1.0f );
	return K;
}

inline cuda_function_header float34 float_camera_intrinsic_extrinsic_to_P( float_camera_intrinsic ci, float_camera_extrinsic ce )
{
	float34	P;
	int		i;
	for( i=0; i<3; i++ ) {
		P.p[i].x = ( ci.fc.x * ce.R.p[i].x + ci.cc.x * ce.R.p[i].z );
		P.p[i].y = ( ci.fc.y * ce.R.p[i].y + ci.cc.y * ce.R.p[i].z );
		P.p[i].z = ( ce.R.p[i].z );
	}
	P.p[3].x = ( ci.fc.x * ce.t.x + ci.cc.x * ce.t.z );
	P.p[3].y = ( ci.fc.y * ce.t.y + ci.cc.y * ce.t.z );
	P.p[3].z = ( ce.t.z );
	return P;
}

///////////////////////////////
// cu_geometry_functions.cpp //
///////////////////////////////

void float_camera_extrinsic_print( float_camera_extrinsic c_ext );

/// N point registration: direct method to compute R and T
/// p2_i = R * p1_i + T, i=1,...,N
float_camera_extrinsic float_camera_extrinsic_relative_from_point_pairs( float3 *p1_ptr, float3 *p2_ptr, int size );

// You should check these parameters before using the following functions.
// d_t:			distance threshold to verity inliers (e.g. 0.005f)
// size:		# of corresponding points (e.g. >3 or >9)
// inliers_n:	# of detected inliers, i.e., if it is large, it means higher confidence.

/// Find rigid transform (coarse) from n points correnpondence using dRMS (distance root mean squared error)
/// p2_i = R * p1_i + T, i=1,...,N
float_camera_extrinsic float_camera_extrinsic_relative_from_point_pairs_drms_coarse( float3 *p1_ptr, float3 *p2_ptr, int size );
float_camera_extrinsic float_camera_extrinsic_relative_from_point_pairs_drms( float3 *p1_ptr, float3 *p2_ptr, int size, float d_t );

int float_camera_extrinsic_relative_get_inliers_crms(
	float_camera_extrinsic t_r, float3 *p1_ptr, float3 *p2_ptr, uchar *inliers_ptr, int size, float d_t );

float_camera_extrinsic float_camera_extrinsic_relative_sparse_bundle_adjustment( float_camera_intrinsic c_int,
	float_camera_extrinsic t_r, float2 *points1, float2 *points2, float3 *vertex2, int size );

/// Find rigid transform (coarse to fine) from n points correnpondence using a combined approach (dRMS+DT)
/// p2_i = R * p1_i + T, i=1,...,N
float_camera_extrinsic float_camera_extrinsic_relative_from_point_pairs_combined(
	float3 *h_point3d_1, float3 *h_point3d_2, int size, float d_t, int *inliers_n );

/// Find rigid transform (coarse to fine) from n points correnpondence using a combined approach (dRMS+BA)
/// p2_i = R * p1_i + T, i=1,...,N
float_camera_extrinsic float_camera_extrinsic_relative_from_point_pairs_combined( 
	float_camera_intrinsic c1_int, float2 *h_point2d_1, float3 *h_point3d_1,
	float_camera_intrinsic c2_int, float2 *h_point2d_2, float3 *h_point3d_2, int size, float d_t, int *inliers_n );

#endif /* !_CU_GEOMETRY_FUNCTIONS_H */

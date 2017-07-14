/**
	\file		cu_vector_functions.h
	\brief		vector functions
	\author		Seongoh Lee
	\version	1.0
	\date		2011.12.09
*/

#ifndef _CU_VECTOR_FUNCTIONS_H
#define _CU_VECTOR_FUNCTIONS_H

#include "cu_device.h"
#include "cu_math_functions.h"

//////////////
// Vector 2 //
//////////////

#define vector2_set(a,v)		{ (a).x=(v); (a).y=(v); } ///< a_i = v, i=1,2
#define vector2_copy(c,a)		{ (c).x=(a).x; (c).y=(a).y; } ///< c_i = a_i, i=1,2
#define vector2_negation(c,a)	{ (c).x=-(a).x; (c).y=-(a).y; } ///< c_i = -a_i, i=1,2

#define vector2_make_inhomogeneous(c,a)	{ (c).x=(a).x/(a).z; (c).y=(a).y/(a).z; } ///< get an inhomogeneous point from a homogeneous point
#define vector2_make_homogeneous(c,a,v)	{ (c).x=(a).x*(v); (c).y=(a).y*(v); (c).z=(v); } ///< get a homogeneous point from an inhomogeneous point

#define vector2_add(c,a,b)		{ c.x=a.x+b.x; c.y=a.y+b.y; } ///< c_i = a_i + b_i, i=1,2
#define vector2_sub(c,a,b)		{ c.x=a.x-b.x; c.y=a.y-b.y; } ///< c_i = a_i - b_i, i=1,2
#define vector2_mul(c,a,b)		{ c.x=a.x*b.x; c.y=a.y*b.y; } ///< c_i = a_i * b_i, i=1,2
#define vector2_div(c,a,b)		{ c.x=a.x/b.x; c.y=a.y/b.y; } ///< c_i = a_i / b_i, i=1,2

#define vector2_add_c(c,a,v)	{ (c).x=(a).x+(v); (c).y=(a).y+(v); } ///< c_i = a_i + v, i=1,2
#define vector2_sub_c(c,a,v)	{ (c).x=(a).x-(v); (c).y=(a).y-(v); } ///< c_i = a_i - v, i=1,2
#define vector2_mul_c(c,a,v)	{ (c).x=(a).x*(v); (c).y=(a).y*(v); } ///< c_i = a_i * v, i=1,2
#define vector2_div_c(c,a,v)	{ (c).x=(a).x/(v); (c).y=(a).y/(v); } ///< c_i = a_i / v, i=1,2

#define vector2_dot(a,b)		( (a).x*(b).x+(a).y*(b).y ) ///< dot product of two vectors (sigma(a_i*b_i), i=1,2)
#define vector2_sum_sq(a)		( vector1_sq((a).x)+vector1_sq((a).y) ) ///< sigma(a_i^2), i=1,2
//template<class T, class S> inline cuda_function_header S vector2_sum_sq( const T &a ) { return( vector1_sq((a).x)+vector1_sq((a).y); } ///< sigma(a_i^2), i=1,2

#define float2_norm2(a)			sqrtf(vector2_sum_sq(a)) ///< L2 Norm
#define float2_distance(a,b)	sqrtf(float2_sum_sq_diff(a,b)) ///< Euclidean distance (dim=2)

inline cuda_function_header float2 float2_make_inhomogeneous( float3 p )
{
	float2	q; vector2_make_inhomogeneous(q,p); return q;
}

inline cuda_function_header float3 float2_make_homogeneous( float2 p, float v )
{
	float3	q; vector2_make_homogeneous(q,p,v); return q;
}

inline cuda_function_header float2 float2_set( float v )
{
	float2 c; vector2_set(c,v); return c;
}

inline cuda_function_header float2 float2_negation(float2 a)
{
	float2 c; vector2_negation(c,a); return c;
}

inline cuda_function_header float2 float2_add(float2 a, float2 b)
{
	float2 c; vector2_add(c,a,b); return c;
}

inline cuda_function_header float2 float2_sub(float2 a, float2 b)
{
	float2 c; vector2_sub(c,a,b); return c;
}

inline cuda_function_header float2 float2_mul(float2 a, float2 b)
{
	float2 c; vector2_mul(c,a,b); return c;
}

inline cuda_function_header float2 float2_div(float2 a, float2 b)
{
	float2 c; vector2_div(c,a,b); return c;
}

inline cuda_function_header float2 float2_add_c(float2 a, float v)
{
	float2 c; vector2_add_c(c,a,v); return c;
}

inline cuda_function_header float2 float2_sub_c(float2 a, float v)
{
	float2 c; vector2_sub_c(c,a,v); return c;
}

inline cuda_function_header float2 float2_mul_c(float2 a, float v)
{
	float2 c; vector2_mul_c(c,a,v); return c;
}

inline cuda_function_header float2 float2_div_c(float2 a, float v)
{
	float2 c; vector2_div_c(c,a,v); return c;
}

inline cuda_function_header float float2_dot(float2 a, float2 b)
{
	return vector2_dot(a,b);
}

inline cuda_function_header float float2_sum_sq(float2 a)
{
	return vector2_sum_sq(a);
}

inline cuda_function_header float float2_sum_sq_diff(float2 a, float2 b)
{
	float	diff = a.x - b.x;
	float	sum = vector1_sq( diff );
	diff = a.y - b.y; sum += vector1_sq( diff );
	return sum;
}

inline cuda_function_header float2 float2_unit(float2 a)
{
	float norm = float2_norm2(a);
	if( float1_non_zero(norm) )
		a = float2_div_c( a, norm );
	return a;
}

//////////////
// Vector 3 //
//////////////

#define vector3_print( v ) ( printf("\n%g %g %g\n", (v).x, (v).y, (v).z) )

#define vector3_set(a,v)		{ (a).x=(v); (a).y=(v); (a).z=(v); } ///< a_i = v, i=1,2,3
#define vector3_copy(c,a)		{ (c).x=(a).x; (c).y=(a).y; (c).z=(a).z; } ///< c_i = a_i, i=1,2,3
#define vector3_negation(c,a)	{ (c).x=-(a).x; (c).y=-(a).y; (c).z=-(a).z; } ///< c_i = -a_i, i=1,2,3

#define vector3_make_inhomogeneous(c,a)	{ (c).x=(a).x/(a).w; (c).y=(a).y/(a).w; (c).z=(a).z/(a).w; } ///< get an inhomogeneous point from a homogeneous point
#define vector3_make_homogeneous(c,a,v)	{ (c).x=(a).x*(v); (c).y=(a).y*(v); (c).z=(a).z*(v); (c).w=(v); } ///< get a homogeneous point from an inhomogeneous point

#define vector3_add(c,a,b)	{ c.x=a.x+b.x; c.y=a.y+b.y; c.z=a.z+b.z; } ///< c_i = a_i + b_i, i=1,2,3
#define vector3_sub(c,a,b)	{ c.x=a.x-b.x; c.y=a.y-b.y; c.z=a.z-b.z; } ///< c_i = a_i - b_i, i=1,2,3
#define vector3_mul(c,a,b)	{ c.x=a.x*b.x; c.y=a.y*b.y; c.z=a.z*b.z; } ///< c_i = a_i - b_i, i=1,2,3
#define vector3_div(c,a,b)	{ c.x=a.x/b.x; c.y=a.y/b.y; c.z=a.z/b.z; } ///< c_i = a_i - b_i, i=1,2,3

#define vector3_add_c(c,a,v)	{ (c).x=(a).x+(v); (c).y=(a).y+(v); (c).z=(a).z+(v); } ///< c_i = a_i + v, i=1,2,3
#define vector3_sub_c(c,a,v)	{ (c).x=(a).x-(v); (c).y=(a).y-(v); (c).z=(a).z-(v); } ///< c_i = a_i - v, i=1,2,3
#define vector3_mul_c(c,a,v)	{ (c).x=(a).x*(v); (c).y=(a).y*(v); (c).z=(a).z*(v); } ///< c_i = a_i * v, i=1,2,3
#define vector3_div_c(c,a,v)	{ (c).x=(a).x/(v); (c).y=(a).y/(v); (c).z=(a).z/(v); } ///< c_i = a_i / v, i=1,2,3

#define vector3_abs(c,a)		{ (c).x=abs((a).x); (c).y=abs((a).y); (c).z=abs((a).z); } ///< c_i = abs( a_i ), i=1,2,3
#define vector3_dot(a,b)		( (a).x*(b).x+(a).y*(b).y+(a).z*(b).z ) ///< dot product of two vectors (sigma(a_i*b_i), i=1,2,3)
#define vector3_cross(c,a,b)	{ (c).x=((a).y*(b).z-(a).z*(b).y); (c).y=((a).z*(b).x-(a).x*(b).z); \
								  (c).z=((a).x*(b).y-(a).y*(b).x); } ///< cross product: c = a ^ b
#define vector3_add_sq(c,a,b)	{ (c).x=vector1_sq((a).x)+vector1_sq((b).x); (c).y=vector1_sq((a).y)+vector1_sq((b).y); \
								  (c).z=vector1_sq((a).z)+vector1_sq((b).z); } ///< c_i = sq(a_i)+sq(b_i), i=1,2,3
#define vector3_sum_sq(a)		( vector1_sq((a).x)+vector1_sq((a).y)+vector1_sq((a).z) ) ///< sigma(a_i^2), i=1,2,3

//template<class T> inline cuda_function_header void vector3_add_sq( T &c, const T &a, const T &b )
//{
//	c.x = vector1_sq(a.x) + vector1_sq(b.x);
//	c.y = vector1_sq(a.y) + vector1_sq(b.y);
//	c.z = vector1_sq(a.z) + vector1_sq(b.z);
//}
//template<class T, class S> inline cuda_function_header S vector3_sum_sq( const T &a) { return (vector1_sq(a.x)+vector1_sq(a.y)+vector1_sq(a.z)); } ///< sigma(a_i^2), i=1,2,3

#define vector3_add_mul_c(c,a,v)	{ (c).x+=(a).x*(v); (c).y+=(a).y*(v); (c).z+=(a).z*(v); } ///< c_i += a_i * v, i=1,2,3

#define float3_sqrt(c,a)		{ (c).x=sqrtf((a).x); (c).y=sqrtf((a).y); (c).z=sqrtf((a).z); } ///< c_i = sqrt( a_i ), i=1,2,3
#define float3_norm2(a)			sqrtf(vector3_sum_sq(a)) ///< L2 Norm
#define float3_distance(a,b)	sqrtf(float3_sum_sq_diff(a,b)) ///< Euclidean distance (dim=2)

inline cuda_function_header float3 float3_set( float v )
{
	float3 c; vector3_set(c,v); return c;
}

inline cuda_function_header float3 float3_negation(float3 a)
{
	float3 c; vector3_negation(c,a); return c;
}

inline cuda_function_header float3 float3_make_inhomogeneous( float4 p )
{
	float3	q; vector3_make_inhomogeneous(q,p); return q;
}

inline cuda_function_header float4 float3_make_homogeneous( float3 p, float v )
{
	float4	q; vector3_make_homogeneous(q,p,v); return q;
}

inline cuda_function_header float3 float3_add(float3 a, float3 b)
{
	float3 c; vector3_add(c,a,b); return c;
}

inline cuda_function_header float3 float3_sub(float3 a, float3 b)
{
	float3 c; vector3_sub(c,a,b); return c;
}

inline cuda_function_header float3 float3_mul(float3 a, float3 b)
{
	float3 c; vector3_mul(c,a,b); return c;
}

inline cuda_function_header float3 float3_div(float3 a, float3 b)
{
	float3 c; vector3_div(c,a,b); return c;
}

inline cuda_function_header float3 float3_add_c(float3 a, float v)
{
	float3 c; vector3_add_c(c,a,v); return c;
}

inline cuda_function_header float3 float3_sub_c(float3 a, float v)
{
	float3 c; vector3_sub_c(c,a,v); return c;
}

inline cuda_function_header float3 float3_mul_c(float3 a, float v)
{
	float3 c; vector3_mul_c(c,a,v); return c;
}

inline cuda_function_header float3 float3_div_c(float3 a, float v)
{
	float3 c; vector3_div_c(c,a,v); return c;
}

inline cuda_function_header float float3_dot(float3 a, float3 b)
{
	return vector3_dot(a,b);
}

inline cuda_function_header float float3_sum_sq(float3 a)
{
	return vector3_sum_sq(a);
}

inline cuda_function_header float3 float3_cross(float3 a, float3 b)
{
	float3 c; vector3_cross(c,a,b); return c;
}

inline cuda_function_header float float3_sum_sq_diff(float3 a, float3 b)
{
	float	diff = a.x - b.x;
	float	sum = vector1_sq( diff );
	diff = a.y - b.y; sum += vector1_sq( diff );
	diff = a.z - b.z; sum += vector1_sq( diff );
	return sum;
}

inline cuda_function_header float3 float3_unit(float3 a)
{
	float	norm = float3_norm2(a);
	if( float1_non_zero(norm) )
		vector3_div_c( a, a, norm );
	return a;
}

// dRMS^2 (distance root mean squared error) = ( ||p1-p2|| - ||q1-q2|| )^2
inline cuda_function_header float float3_sq_drms( float3 p1, float3 p2, float3 q1, float3 q2 )
{
	return vector1_sq( float3_distance(p1,p2) - float3_distance(q1,q2) );
}

//////////////
// Vector 4 //
//////////////

#define vector4_set(a,v)		{ (a).x=(v); (a).y=(v); (a).z=(v); (a).w=(v); } ///< a_i = v, i=1,2,3,4
#define vector4_copy(c,a)		{ (c).x=(a).x; (c).y=(a).y; (c).z=(a).z; (c).w=(a).w; } ///< c_i = a_i, i=1,2,3,4
#define vector4_negation(c,a)	{ (c).x=-(a).x; (c).y=-(a).y; (c).z=-(a).z; (c).w=-(a).w; } ///< c_i = -a_i, i=1,2,3,4

#define vector4_add(c,a,b)	{ c.x=a.x+b.x; c.y=a.y+b.y; c.z=a.z+b.z; c.w=a.w+b.w; } ///< c_i = a_i + b_i, i=1,2,3,4
#define vector4_sub(c,a,b)	{ c.x=a.x-b.x; c.y=a.y-b.y; c.z=a.z-b.z; c.w=a.w-b.w; } ///< c_i = a_i - b_i, i=1,2,3,4
#define vector4_mul(c,a,b)	{ c.x=a.x*b.x; c.y=a.y*b.y; c.z=a.z*b.z; c.w=a.w*b.w; } ///< c_i = a_i - b_i, i=1,2,3,4
#define vector4_div(c,a,b)	{ c.x=a.x/b.x; c.y=a.y/b.y; c.z=a.z/b.z; c.w=a.w/b.w; } ///< c_i = a_i - b_i, i=1,2,3,4

#define vector4_add_c(c,a,v)	{ (c).x=(a).x+(v); (c).y=(a).y+(v); (c).z=(a).z+(v); (c).w=(a).w+(v); } ///< c_i = a_i + v, i=1,2,3,4
#define vector4_sub_c(c,a,v)	{ (c).x=(a).x-(v); (c).y=(a).y-(v); (c).z=(a).z-(v); (c).w=(a).w-(v); } ///< c_i = a_i - v, i=1,2,3,4
#define vector4_mul_c(c,a,v)	{ (c).x=(a).x*(v); (c).y=(a).y*(v); (c).z=(a).z*(v); (c).w=(a).w*(v); } ///< c_i = a_i * v, i=1,2,3,4
#define vector4_div_c(c,a,v)	{ (c).x=(a).x/(v); (c).y=(a).y/(v); (c).z=(a).z/(v); (c).w=(a).w/(v); } ///< c_i = a_i / v, i=1,2,3,4

#define vector4_dot(a,b)		( (a).x*(b).x+(a).y*(b).y+(a).z*(b).z+(a).w*(b).w ) ///< dot product of two vectors (sigma(a_i*b_i), i=1,2,3,4)
#define vector4_sum_sq(a)		( vector1_sq((a).x)+vector1_sq((a).y)+vector1_sq((a).z)+vector1_sq((a).w) ) ///< sigma(a_i^2), i=1,2,3,4

#define float4_norm2(a)			sqrtf(vector4_sum_sq(a)) ///< L2 Norm
#define float4_distance(a,b)	sqrtf(float4_sum_sq_diff(a,b)) ///< Euclidean distance (dim=2)

inline cuda_function_header float4 float4_set( float v )
{
	float4 c; vector4_set(c,v); return c;
}

inline cuda_function_header float4 float4_negation(float4 a)
{
	float4 c; vector4_negation(c,a); return c;
}

inline cuda_function_header float4 float4_add(float4 a, float4 b)
{
	float4 c; vector4_add(c,a,b); return c;
}

inline cuda_function_header float4 float4_sub(float4 a, float4 b)
{
	float4 c; vector4_sub(c,a,b); return c;
}

inline cuda_function_header float4 float4_mul(float4 a, float4 b)
{
	float4 c; vector4_mul(c,a,b); return c;
}

inline cuda_function_header float4 float4_div(float4 a, float4 b)
{
	float4 c; vector4_div(c,a,b); return c;
}

inline cuda_function_header float4 float4_add_c(float4 a, float v)
{
	float4 c; vector4_add_c(c,a,v); return c;
}

inline cuda_function_header float4 float4_sub_c(float4 a, float v)
{
	float4 c; vector4_sub_c(c,a,v); return c;
}

inline cuda_function_header float4 float4_mul_c(float4 a, float v)
{
	float4 c; vector4_mul_c(c,a,v); return c;
}

inline cuda_function_header float4 float4_div_c(float4 a, float v)
{
	float4 c; vector4_div_c(c,a,v); return c;
}

inline cuda_function_header float float4_dot(float4 a, float4 b)
{
	return vector4_dot(a,b);
}

inline cuda_function_header float float4_sum_sq(float4 a)
{
	return vector4_sum_sq(a);
}

inline cuda_function_header float float4_sum_sq_diff(float4 a, float4 b)
{
	float	diff = a.x - b.x;
	float	sum = vector1_sq( diff );
	diff = a.y - b.y; sum += vector1_sq( diff );
	diff = a.z - b.z; sum += vector1_sq( diff );
	diff = a.w - b.w; sum += vector1_sq( diff );
	return sum;
}

inline cuda_function_header float4 float4_unit(float4 a)
{
	float norm = float4_norm2(a);
	if( float1_non_zero(norm) )
		a = float4_div_c( a, norm );
	return a;
}

////////////////////////
// Matrix definitions //
////////////////////////

typedef struct _float22
{
	float2	p[2];
} float22;

typedef struct _float33
{
	float3	p[3];
} float33;

typedef struct _float34
{
	float3	p[4];
} float34;

////////////////////////////////////////////
// Matrix 2x2: float2 m[2] (column-major) //
////////////////////////////////////////////

#define matrix22_add_c(c,a,v)	{ vector2_add_c(c[0],a[0],v) vector2_add_c(c[1],a[1],v) }
#define matrix22_sub_c(c,a,v)	{ vector2_sub_c(c[0],a[0],v) vector2_sub_c(c[1],a[1],v) }
#define matrix22_mul_c(c,a,v)	{ vector2_mul_c(c[0],a[0],v) vector2_mul_c(c[1],a[1],v) }
#define matrix22_div_c(c,a,v)	{ vector2_div_c(c[0],a[0],v) vector2_div_c(c[1],a[1],v) }

#define matrix22_set(a,v) { (a)[0].x=(v); (a)[1].x=(v); (a)[0].y=(v); (a)[1].y=(v); }
#define matrix22_identity(a) { (a)[0].x=1; (a)[1].x=0; (a)[0].y=0; (a)[1].y=1; } ///< 2x2 identity matrix
#define matrix22_det(a) ( (a)[0].x * (a)[1].y - (a)[1].x * (a)[0].y ) ///< the determinant of 2x2 matrix

#define matrix22_mm(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[1].x*(b)[0].y; (c)[0].y = (a)[0].y*(b)[0].x + (a)[1].y*(b)[0].y; \
	(c)[1].x = (a)[0].x*(b)[1].x + (a)[1].x*(b)[1].y; (c)[1].y = (a)[0].y*(b)[1].x + (a)[1].y*(b)[1].y; } ///< c [2 x 2] = a [2 x 2] * b [2 x 2] (column-major)
#define matrix22_mt(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[1].x*(b)[1].x; (c)[0].y = (a)[0].y*(b)[0].x + (a)[1].y*(b)[1].x; \
	(c)[1].x = (a)[0].x*(b)[0].y + (a)[1].x*(b)[1].y; (c)[1].y = (a)[0].y*(b)[0].y + (a)[1].y*(b)[1].y; } ///< c [2 x 2] = a [2 x 2] * b^T [2 x 2] (column-major)
#define matrix22_tm(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[0].y*(b)[0].y; (c)[0].y = (a)[1].x*(b)[0].x + (a)[1].y*(b)[0].y; \
	(c)[1].x = (a)[0].x*(b)[1].x + (a)[0].y*(b)[1].y; (c)[1].y = (a)[1].x*(b)[1].x + (a)[1].y*(b)[1].y; } ///< c [2 x 2] = a^T [2 x 2] * b [2 x 2] (column-major)
#define matrix22_tt(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[0].y*(b)[1].x; (c)[0].y = (a)[1].x*(b)[0].x + (a)[1].y*(b)[1].x; \
	(c)[1].x = (a)[0].x*(b)[0].y + (a)[0].y*(b)[1].y; (c)[1].y = (a)[1].x*(b)[0].y + (a)[1].y*(b)[1].y; } ///< c [2 x 2] = a^T [2 x 2] * b^T [2 x 2] (column-major)
#define matrix22_mv(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[1].x*(v).y; (c).y=(a)[0].y*(v).x+(a)[1].y*(v).y; } ///< c[1].x=m)[2x2]*v[1].x (column-major) or c[1].x=m^t[2x2]*v[1].x (row-major)
#define matrix22_tv(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[0].y*(v).y; (c).y=(a)[1].x*(v).x+(a)[1].y*(v).y; } ///< c[1].x=m^t[2x2]*v[1].x (column-major) or c[1].x=m[2x2]*v[1].x (row-major)

inline cuda_function_header float22 float22_set( float v )
{
	float22	c; matrix22_set( c.p, v ) return c;
}

inline cuda_function_header float22 float22_add_c( float22 a, float v)
{
	float22 c; matrix22_add_c(c.p,a.p,v) return c;
}

inline cuda_function_header float22 float22_sub_c( float22 a, float v)
{
	float22 c; matrix22_sub_c(c.p,a.p,v) return c;
}

inline cuda_function_header float22 float22_mul_c( float22 a, float v)
{
	float22 c; matrix22_mul_c(c.p,a.p,v) return c;
}

inline cuda_function_header float22 float22_div_c( float22 a, float v)
{
	float22 c; matrix22_div_c(c.p,a.p,v) return c;
}

inline cuda_function_header float22 float22_identity()
{
	float22	c; matrix22_identity( c.p ) return c;
}

inline cuda_function_header bool float22_inv( float22 *c, float22 a )
{
	float	det = matrix22_det( a.p );
	if( float1_zero(det) ) return false;
	c->p[0].x = a.p[1].y;	c->p[1].x = - a.p[1].x;
	c->p[0].y = - a.p[0].y;	c->p[1].y = a.p[0].x;
	matrix22_div_c( c->p, a.p, det );
	return true;
}

inline cuda_function_header float22 float22_mm( float22 a, float22 b )
{
	float22	c;	matrix22_mm( c.p, a.p, b.p ) return c;
}

inline cuda_function_header float22 float22_mt( float22 a, float22 b )
{
	float22	c;	matrix22_mt( c.p, a.p, b.p ) return c;
}
inline cuda_function_header float22 float22_tm( float22 a, float22 b )
{
	float22	c;	matrix22_tm( c.p, a.p, b.p ) return c;
}

inline cuda_function_header float22 float22_tt( float22 a, float22 b )
{
	float22	c;	matrix22_tt( c.p, a.p, b.p ) return c;
}

inline cuda_function_header float2 float22_mv( float22 a, float2 v ) 
{
	float2	c; matrix22_mv( c, a.p, v ) return c;
}

inline cuda_function_header float2 float22_tv( float22 a, float2 v ) 
{
	float2	c; matrix22_tv( c, a.p, v ) return c;
}

////////////////////////////////////////////
// Matrix 3x3: float3 m[3] (column-major) //
////////////////////////////////////////////

#define matrix33_print( m ) (	printf("\n%g %g %g\n%g %g %g\n%g %g %g\n", (m).p[0].x, (m).p[1].x, (m).p[2].x, \
								(m).p[0].y, (m).p[1].y, (m).p[2].y, (m).p[0].z, (m).p[1].z, (m).p[2].z) )

#define matrix33_to_vector( m, v ) { v[0] = m.p[0].x; v[3] = m.p[1].x; v[6] = m.p[2].x; \
									 v[1] = m.p[0].y; v[4] = m.p[1].y; v[7] = m.p[2].y; \
									 v[2] = m.p[0].z; v[5] = m.p[1].z; v[8] = m.p[2].z; }
#define vector_to_matrix33( v, m ) { m.p[0].x = v[0]; m.p[1].x = v[3]; m.p[2].x = v[6]; \
									 m.p[0].y = v[1]; m.p[1].y = v[4]; m.p[2].y = v[7]; \
									 m.p[0].z = v[2]; m.p[1].z = v[5]; m.p[2].z = v[8]; }

#define matrix33_add_c(c,a,v)	{ vector3_add_c(c[0],a[0],v) vector3_add_c(c[1],a[1],v) vector3_add_c(c[2],a[2],v) }
#define matrix33_sub_c(c,a,v)	{ vector3_sub_c(c[0],a[0],v) vector3_sub_c(c[1],a[1],v) vector3_sub_c(c[2],a[2],v) }
#define matrix33_mul_c(c,a,v)	{ vector3_mul_c(c[0],a[0],v) vector3_mul_c(c[1],a[1],v) vector3_mul_c(c[2],a[2],v) }
#define matrix33_div_c(c,a,v)	{ vector3_div_c(c[0],a[0],v) vector3_div_c(c[1],a[1],v) vector3_div_c(c[2],a[2],v) }

#define matrix33_identity(a) { 	(a)[0].x=1; (a)[0].y=0; (a)[0].z=0; \
								(a)[1].x=0; (a)[1].y=1; (a)[1].z=0; \
								(a)[2].x=0; (a)[2].y=0; (a)[2].z=1; } ///< 3x3 identity matrix

#define matrix33_t(c,a)	{ (c)[0].x=(a)[0].x;(c)[0].y=(a)[1].x;(c)[0].z=(a)[2].x;(c)[1].x=(a)[0].y;(c)[1].y=(a)[1].y;(c)[1].z=(a)[2].y; \
						  (c)[2].x=(a)[0].z;(c)[2].y=(a)[1].z;(c)[2].z=(a)[2].z; } ///< transpose 3x3 matrix a to c

#define matrix33_det(a) ( ((a)[0].x*((a)[2].z*(a)[1].y-(a)[2].y*(a)[1].z)-(a)[1].x*((a)[2].z*(a)[0].y-(a)[2].y*(a)[0].z)+ \
						 (a)[2].x*((a)[1].z*(a)[0].y-(a)[1].y*(a)[0].z)) ) ///< the determinant of 3x3 matrix (A matrix and its transpose have the same determinant.)

#define matrix33_skew_symmetric(a,v) { (a)[0].x=(a)[1].y=(a)[2].z=0.0f;(a)[0].y=(v).z;(a)[1].x=-(v).z;(a)[0].z=-(v).y;(a)[2].x=(v).y; \
	(a)[1].z=(v).x;(a)[2].y=-(v).x; } ///< get skew-symmetric matrix(3x3) from a vector(3)

#define matrix33_mm(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[1].x*(b)[0].y + (a)[2].x*(b)[0].z; (c)[0].y = (a)[0].y*(b)[0].x + (a)[1].y*(b)[0].y + (a)[2].y*(b)[0].z; \
	(c)[0].z = (a)[0].z*(b)[0].x + (a)[1].z*(b)[0].y + (a)[2].z*(b)[0].z; (c)[1].x = (a)[0].x*(b)[1].x + (a)[1].x*(b)[1].y + (a)[2].x*(b)[1].z; (c)[1].y = (a)[0].y*(b)[1].x + (a)[1].y*(b)[1].y + (a)[2].y*(b)[1].z; \
	(c)[1].z = (a)[0].z*(b)[1].x + (a)[1].z*(b)[1].y + (a)[2].z*(b)[1].z; (c)[2].x = (a)[0].x*(b)[2].x + (a)[1].x*(b)[2].y + (a)[2].x*(b)[2].z; (c)[2].y = (a)[0].y*(b)[2].x + (a)[1].y*(b)[2].y + (a)[2].y*(b)[2].z; \
	(c)[2].z = (a)[0].z*(b)[2].x + (a)[1].z*(b)[2].y + (a)[2].z*(b)[2].z; } ///< c [3 x 3] = a [3 x 3] * b [3 x 3] (column-major)

#define matrix33_mt(c,a,b) {  (c)[0].x = (a)[0].x*(b)[0].x + (a)[1].x*(b)[1].x + (a)[2].x*(b)[2].x; (c)[0].y = (a)[0].y*(b)[0].x + (a)[1].y*(b)[1].x + (a)[2].y*(b)[2].x; \
	(c)[0].z = (a)[0].z*(b)[0].x + (a)[1].z*(b)[1].x + (a)[2].z*(b)[2].x; (c)[1].x = (a)[0].x*(b)[0].y + (a)[1].x*(b)[1].y + (a)[2].x*(b)[2].y; (c)[1].y = (a)[0].y*(b)[0].y + (a)[1].y*(b)[1].y + (a)[2].y*(b)[2].y; \
	(c)[1].z = (a)[0].z*(b)[0].y + (a)[1].z*(b)[1].y + (a)[2].z*(b)[2].y; (c)[2].x = (a)[0].x*(b)[0].z + (a)[1].x*(b)[1].z + (a)[2].x*(b)[2].z; (c)[2].y = (a)[0].y*(b)[0].z + (a)[1].y*(b)[1].z + (a)[2].y*(b)[2].z; \
	(c)[2].z = (a)[0].z*(b)[0].z + (a)[1].z*(b)[1].z + (a)[2].z*(b)[2].z; } ///< c [3 x 3] = a [3 x 3] * b^T [3 x 3] (column-major)

#define matrix33_tm(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[0].y*(b)[0].y + (a)[0].z*(b)[0].z; (c)[0].y = (a)[1].x*(b)[0].x + (a)[1].y*(b)[0].y + (a)[1].z*(b)[0].z; \
	(c)[0].z = (a)[2].x*(b)[0].x + (a)[2].y*(b)[0].y + (a)[2].z*(b)[0].z; (c)[1].x = (a)[0].x*(b)[1].x + (a)[0].y*(b)[1].y + (a)[0].z*(b)[1].z; (c)[1].y = (a)[1].x*(b)[1].x + (a)[1].y*(b)[1].y + (a)[1].z*(b)[1].z; \
	(c)[1].z = (a)[2].x*(b)[1].x + (a)[2].y*(b)[1].y + (a)[2].z*(b)[1].z; (c)[2].x = (a)[0].x*(b)[2].x + (a)[0].y*(b)[2].y + (a)[0].z*(b)[2].z; (c)[2].y = (a)[1].x*(b)[2].x + (a)[1].y*(b)[2].y + (a)[1].z*(b)[2].z; \
	(c)[2].z = (a)[2].x*(b)[2].x + (a)[2].y*(b)[2].y + (a)[2].z*(b)[2].z; } ///< c [3 x 3] = a^T [3 x 3] * b [3 x 3] (column-major)

#define matrix33_tt(c,a,b) { (c)[0].x = (a)[0].x*(b)[0].x + (a)[0].y*(b)[1].x + (a)[0].z*(b)[2].x; (c)[0].y = (a)[1].x*(b)[0].x + (a)[1].y*(b)[1].x + (a)[1].z*(b)[2].x; \
	(c)[0].z = (a)[2].x*(b)[0].x + (a)[2].y*(b)[1].x + (a)[2].z*(b)[2].x; (c)[1].x = (a)[0].x*(b)[0].y + (a)[0].y*(b)[1].y + (a)[0].z*(b)[2].y; (c)[1].y = (a)[1].x*(b)[0].y + (a)[1].y*(b)[1].y + (a)[1].z*(b)[2].y; \
	(c)[1].z = (a)[2].x*(b)[0].y + (a)[2].y*(b)[1].y + (a)[2].z*(b)[2].y; (c)[2].x = (a)[0].x*(b)[0].z + (a)[0].y*(b)[1].z + (a)[0].z*(b)[2].z; (c)[2].y = (a)[1].x*(b)[0].z + (a)[1].y*(b)[1].z + (a)[1].z*(b)[2].z; \
	(c)[2].z = (a)[2].x*(b)[0].z + (a)[2].y*(b)[1].z + (a)[2].z*(b)[2].z; } ///< c [3 x 3] = a^T [3 x 3] * b^T [3 x 3] (column-major)

#define matrix33_mv(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[1].x*(v).y+(a)[2].x*(v).z; (c).y=(a)[0].y*(v).x+(a)[1].y*(v).y+(a)[2].y*(v).z; \
	(c).z=(a)[0].z*(v).x+(a)[1].z*(v).y+(a)[2].z*(v).z; } ///< c[3]=m[3x3]*v[3] (column-major) or c[3]=m^t[3x3]*v[3] (row-major)

#define matrix33_tv(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[0].y*(v).y+(a)[0].z*(v).z; (c).y=(a)[1].x*(v).x+(a)[1].y*(v).y+(a)[1].z*(v).z; \
	(c).z=(a)[2].x*(v).x+(a)[2].y*(v).y+(a)[2].z*(v).z; } ///< c[3]=m^t[3x3]*v[3] (column-major) or c[3]=m[3x3]*v[3] (row-major)

#define matrix33_mv_2(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[1].x*(v).y+(a)[2].x; (c).y=(a)[0].y*(v).x+(a)[1].y*(v).y+(a)[2].y; \
	(c).z=(a)[0].z*(v).x+(a)[1].z*(v).y+(a)[2].z; } ///< c[3]=m[3x3]*v[2|1] (column-major) or c[3]=m^t[3x3]*v[2|1] (row-major)

#define matrix33_tv_2(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[0].y*(v).y+(a)[0].z; (c).y=(a)[1].x*(v).x+(a)[1].y*(v).y+(a)[1].z; \
	(c).z=(a)[2].x*(v).x+(a)[2].y*(v).y+(a)[2].z; } ///< c[3]=m^t[3x3]*v[2|1] (column-major) or c[3]=m[3x3]*v[2|1] (row-major)

/// The inverse of 3x3 matrix

/// \param c		[out] a matrix
/// \param a		[in] a matrix
/// \remarks It works both row- and column-major matrix.
inline cuda_function_header bool float33_inv( float33 *c, float33 a )
{
	float	determinant, A, B, C;
	A = ( a.p[2].z*a.p[1].y - a.p[2].y*a.p[1].z );
	B = ( a.p[0].z*a.p[2].y - a.p[0].y*a.p[2].z );
	C = ( a.p[0].y*a.p[1].z - a.p[0].z*a.p[1].y );
	determinant = ( a.p[0].x * A + a.p[1].x * B + a.p[2].x * C );
	if( float1_zero( determinant ) ) return false;
	c->p[0].x = A;	c->p[0].y = B;	c->p[0].z = C;
	c->p[1].x = ( a.p[1].z*a.p[2].x - a.p[1].x*a.p[2].z ); // D
	c->p[1].y = ( a.p[0].x*a.p[2].z - a.p[0].z*a.p[2].x ); // E
	c->p[1].z = ( a.p[0].z*a.p[1].x - a.p[0].x*a.p[1].z ); // F
	c->p[2].x = ( a.p[1].x*a.p[2].y - a.p[1].y*a.p[2].x ); // G
	c->p[2].y = ( a.p[0].y*a.p[2].x - a.p[0].x*a.p[2].y ); // H
	c->p[2].z = ( a.p[0].x*a.p[1].y - a.p[0].y*a.p[1].x ); // K
	matrix33_div_c( c->p, c->p, determinant );
	return true;
}

inline cuda_function_header float33 float33_identity()
{
	float33	c; matrix33_identity( c.p ); return c;
}

inline cuda_function_header float float33_det( float33 a )
{
	return matrix33_det( a.p );
}

inline cuda_function_header float33 float33_t( float33 a )
{
	float33	c;	matrix33_t( c.p, a.p ) return c;
}

inline cuda_function_header float33 float33_skew_symmetric( float3 v )
{
	float33	c;	matrix33_skew_symmetric( c.p, v ) return c;
}

inline cuda_function_header float33 float33_mm( float33 a, float33 b )
{
	float33	c;	matrix33_mm( c.p, a.p, b.p ) return c;
}

inline cuda_function_header float33 float33_mt( float33 a, float33 b )
{
	float33	c;	matrix33_mt( c.p, a.p, b.p ) return c;
}
inline cuda_function_header float33 float33_tm( float33 a, float33 b )
{
	float33	c;	matrix33_tm( c.p, a.p, b.p ) return c;
}

inline cuda_function_header float33 float33_tt( float33 a, float33 b )
{
	float33	c;	matrix33_tt( c.p, a.p, b.p ) return c;
}

inline cuda_function_header float3 float33_mv( float33 a, float3 v ) 
{
	float3	c; matrix33_mv( c, a.p, v ) return c;
}

inline cuda_function_header float3 float33_tv( float33 a, float3 v ) 
{
	float3	c; matrix33_tv( c, a.p, v ) return c;
}

inline cuda_function_header float3 float33_mv_2( float33 a, float2 v ) 
{
	float3	c; matrix33_mv_2( c, a.p, v ) return c;
}

inline cuda_function_header float3 float33_tv_2( float33 a, float2 v ) 
{
	float3	c; matrix33_tv_2( c, a.p, v ) return c;
}

// 3x3 covariance matrix H = sigma( a b^t )
inline cuda_function_header float33 float33_covariance( float3 *a, float3 *b, int dim ) 
{
	float33	c;
	c.p[0].x = a[0].x * b[0].x; c.p[1].x = a[0].y * b[0].x; c.p[2].x = a[0].z * b[0].x;
	c.p[0].y = a[0].x * b[0].y; c.p[1].y = a[0].y * b[0].y; c.p[2].y = a[0].z * b[0].y;
	c.p[0].z = a[0].x * b[0].z; c.p[1].z = a[0].y * b[0].z; c.p[2].z = a[0].z * b[0].z;
	for( int i=1; i<dim; i++ ) {
		c.p[0].x += a[i].x * b[i].x; c.p[1].x += a[i].y * b[i].x; c.p[2].x += a[i].z * b[i].x;
		c.p[0].y += a[i].x * b[i].y; c.p[1].y += a[i].y * b[i].y; c.p[2].y += a[i].z * b[i].y;
		c.p[0].z += a[i].x * b[i].z; c.p[1].z += a[i].y * b[i].z; c.p[2].z += a[i].z * b[i].z;
	}
	return c;
}

////////////////////////////////////////////
// Matrix 3x4: float3 m[4] (column-major) //
////////////////////////////////////////////

#define matrix34_mv(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[1].x*(v).y+(a)[2].x*(v).z+(a)[3].x*(v).w; \
	(c).y=(a)[0].y*(v).x+(a)[1].y*(v).y+(a)[2].y*(v).z+(a)[3].y*(v).w; \
	(c).z=(a)[0].z*(v).x+(a)[1].z*(v).y+(a)[2].z*(v).z+(a)[3].z*(v).w; } ///< c[3]=m[3x4]*v[4] (column-major)

#define matrix34_tv(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[0].y*(v).y+(a)[0].z*(v).z; \
	(c).y=(a)[1].x*(v).x+(a)[1].y*(v).y+(a)[1].z*(v).z; \
	(c).z=(a)[2].x*(v).x+(a)[2].y*(v).y+(a)[2].z*(v).z; \
	(c).w=(a)[3].x*(v).x+(a)[3].y*(v).y+(a)[3].z*(v).z; } ///< c[4]=m^T[4x3]*v[3] (column-major)

#define matrix34_mv_3(c,a,v) { (c).x=(a)[0].x*(v).x+(a)[1].x*(v).y+(a)[2].x*(v).z+(a)[3].x; \
	(c).y=(a)[0].y*(v).x+(a)[1].y*(v).y+(a)[2].y*(v).z+(a)[3].y; \
	(c).z=(a)[0].z*(v).x+(a)[1].z*(v).y+(a)[2].z*(v).z+(a)[3].z; } ///< c[3]=m[3x4]*v[3|1] (column-major)

inline cuda_function_header float3 float34_mv( float34 a, float4 v ) 
{
	float3	c; matrix34_mv( c, a.p, v ) return c;
}

inline cuda_function_header float4 float34_tv( float34 a, float3 v ) 
{
	float4	c; matrix34_tv( c, a.p, v ) return c;
}

inline cuda_function_header float3 float34_mv_3( float34 a, float3 v ) 
{
	float3	c; matrix34_mv_3( c, a.p, v ) return c;
}

//////////////
// Vector N //
//////////////

/// Copy array

/// \param dst	[out] array
/// \param src	[int] array
/// \param dim	[in] dimension of the array
template<class T> cuda_function_header void vector1s_copy( T *dst, T *src, int dim )
{
	do{
		*dst++ = *src++;
	} while( --dim );
}

/// Fill with a constant

/// \param a	[out] array
/// \param val	[in] a contant
/// \param dim	[in] dimension of the array
template<class T> cuda_function_header void vector1s_fill( T *p, const T val, int dim )
{
	do{
		*p++ = val;
	} while( --dim );
}

/// subtraction
template<class T> cuda_function_header void vector1s_sub( const T *a, const T *b, T *c, int dim )
{
	for( int i=0; i<dim; i++ ) c[i] = ( a[i] - b[i] );
}

/// subtraction (constant)
template<class T> cuda_function_header void vector1s_sub_c( const T *a, const T v, T *c, int dim )
{
	for( int i=0; i<dim; i++ ) c[i] = ( a[i] - v );
}

template<class T> cuda_function_header void vector3s_sub_c( const T *a, const T v, T *c, int dim )
{
	for( int i=0; i<dim; i++ ) vector3_sub( c[i], a[i], v );
}

/// mean
template<class T> cuda_function_header T vector1s_mean( const T *a, int dim )
{
	T	sum = a[0];
	for( int i=1; i<dim; i++ ) sum += a[i];
	return (sum / dim);
}

template<class T> cuda_function_header T vector3s_mean( const T *a, int dim )
{
	T	sum = a[0];
	for( int i=1; i<dim; i++ ) vector3_add( sum, sum, a[i] );
	vector3_div_c( sum, sum, dim );
	return sum;
}

template<class T> cuda_function_header T vector1s_sum_sq( const T *a, int dim )
{
	T	sum = vector1_sq( a[0] );
	for( int i=1; i<dim; i++ ) sum += vector1_sq( a[i] );
	return sum;
}

/// Sum of Squared Differences (SSD)
template<class T> cuda_function_header T vector1s_sum_sq_diff( const T *a, const T *b, int dim )
{
	T	sum = vector1_sq( a[0] - b[0] );
	for( int i=1; i<dim; i++ ) sum += vector1_sq( a[i] - b[i] );
	return sum;
}

/// normalized SSD
cuda_function_header float float1s_nssd( const float *d1, const float*d2, const float v1, const float v2, int dim )
{
	float	sum = vector1s_sum_sq_diff( d1, d2, dim );
	return sum / sqrtf( v1 + v2 ) / 2.f;
}

/// Minimum: m = min(a_i), i=1,...,dim

/// \param a	[in] A
/// \param dim	[in] dimension of the array
/// \return		minimum value
template<class T> cuda_function_header T vector1s_min( T *a, int dim )
{
	int	i;
	T	min = a[0];
	for( i=1; i<dim; i++ ) if( min > a[i] ) min = a[i];
	return min;
}

/// Get the index of minimum element

/// \param a		[in] A
/// \param dim		[in] dimension of the array
/// \return	the minimum index
template<class T> cuda_function_header int vector1s_min_by_key( T *a, int dim )
{
	int		i, j = 0;
	T	min = a[0];
	for( i=1; i<dim; i++ ) if( min > a[i] ) { min = a[i]; j = i; }
	return j;
}

/// Get a vector subset of inliers

/// \param src	[in] source vectors
/// \param dst	[out] inlier vector subset
/// \param mask	[in] inlier mask array
/// \param size	[in] number of source vectors
/// \return number of inliers
template<class T, class U> cuda_function_header int vector1s_get_inlier_subset( T *src, T *dst, U *mask, int size )
{
	int	inliers = 0;
	for( int i=0; i<size; i++ ) {
		if( mask[i] )
			dst[inliers++] = src[i];
	}
	return inliers;
}

#endif /* !_CU_VECTOR_FUNCTIONS_H */

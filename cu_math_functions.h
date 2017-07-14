/**
	\file		cu_math_functions.h
	\brief		math functions
	\author		Seongoh Lee
	\version	1.0
	\date		2011.12.09
*/

#ifndef _CU_MATH_FUNCTIONS_H
#define _CU_MATH_FUNCTIONS_H

#include <helper_math.h>
#include "cu_device.h"

#include <float.h>
#ifndef __CUDACC__
#include <math.h>
#endif

/****************************************************************************************\
*								constant definitions					                 *
\****************************************************************************************/

#define float1_pi	3.141592653589793f
#define float1_pi2	6.283185307179586f
#define float1_pi_2	1.570796326794896f
#define float1_r2d	57.29577951308232f ///< 180/PI
#define float1_d2r	0.017453292519943f ///< PI/180

/****************************************************************************************\
*								Logical definitions						                 *
\****************************************************************************************/

//#define float1_zero(a)		((a)>-FLT_EPSILON && (a)<FLT_EPSILON) 
//#define float1_non_zero(a)	((a)>FLT_EPSILON || (a)<-FLT_EPSILON)
static inline cuda_function_header bool float1_zero(float &a) { return (a>-FLT_EPSILON && a<FLT_EPSILON); }
static inline cuda_function_header bool float1_non_zero(float &a) { return (a>FLT_EPSILON || a<-FLT_EPSILON); }

//#define vector1_max2(a,b)	((a)>(b)?(a):(b)) ///< max operation (a and b)
//#define vector1_max3(a,b,c)	vector1_max2( vector1_max2(a,b), c ) ///< max operation (a, b and c)
//#define vector1_min2(a,b)	((a)<(b)?(a):(b)) ///< min operation (a and b)
//#define vector1_min3(a,b,c)	vector1_min2( vector1_min2(a,b), c ) ///< min operation (a, b and c)
//#define vector1_swap(a,b,t)	{ (t)=(a);(a)=(b);(b)=(t); } ///< general swap operation (all types)
template<class T> inline cuda_function_header T vector1_max2( const T &a, const T &b ) { return (a>b)?a:b; }
template<class T> inline cuda_function_header T vector1_max3( const T &a, const T &b, const T &c ) { return vector1_max2(vector1_max2(a,b),c); }
template<class T> inline cuda_function_header T vector1_min2( const T &a, const T &b ) { return (a<b)?a:b; }
template<class T> inline cuda_function_header T vector1_min3( const T &a, const T &b, const T &c ) { return vector1_min2(vector1_min2(a,b),c); }
template<class T> inline cuda_function_header void vector1_swap( T &a, T &b ) { T t; t=a; a=b; b=t; }

/****************************************************************************************\
*								Mathematical definitions					             *
\****************************************************************************************/

template<class T> inline cuda_function_header T vector1_sq( const T &a ) { return (a*a); } ///< square of a number
//#define vector1_sq(a) ((a)*(a)) ///< square of a number

// Vxy = V00 (1-x)(1-y) + V10 x(1-y) + V01 (1-x)y + V11 xy
template<class T, class U> cuda_function_header T vector3_interpolation_bilinear( T v[4], U x, U y )
{
	U	a = ( 1.0f - x ), b = ( 1.0f - y );
	U	ab = a*b, xb = x*b, ay = a*y, xy = x*y;
	T		u;
	u.x = v[0].x*ab + v[1].x*xb + v[2].x*ay + v[3].x*xy;
	u.y = v[0].y*ab + v[1].y*xb + v[2].y*ay + v[3].y*xy;
	u.z = v[0].z*ab + v[1].z*xb + v[2].z*ay + v[3].z*xy;
	return u;
}

#if defined(__cplusplus)
extern "C" {
#endif

static inline cuda_function_header int int_div_up(int total, int grain)
{
	return (total + grain - 1) / grain;
}

static inline cuda_function_header int float1_round( float a )
{
	return (a < 0) ? (int)(a - 0.5f) : (int)(a + 0.5f);
}

static inline cuda_function_header int2 float2_round( float2 a )
{
	return make_int2( float1_round(a.x), float1_round(a.y) );
}

static inline cuda_function_header uchar int1_to_uchar( float n )
{
	uchar u; if( n>255 ) u = 0xFF; else if( n<0 ) u = 0x00; else u = (uchar)(n); return u;
}

static inline cuda_function_header uchar float1_to_uchar( float f )
{
	uchar u; if( f>255.0F ) u = 0xFF; else if( f<0.0F ) u = 0x00; else u = (uchar)(f); return u;
}

// Vxy = V00 (1-x)(1-y) + V10 x(1-y) + V01 (1-x)y + V11 xy
inline cuda_function_header float float_interpolation_bilinear( float v[4], float x, float y )
{
	float	a = ( 1.0f - x ), b = ( 1.0f - y );
	return ( v[0]*a*b + v[1]*x*b + v[2]*a*y + v[3]*x*y );
}

// Vxyz = V000 (1-x)(1-y)(1-z) + V100 x(1-y)(1-z) + V010 (1-x)y(1-z) +
// V001 (1-x)(1-y)z + V101 x(1-y)z + V011 (1-x)yz + V110 xy(1-z) + V111 xyz
inline cuda_function_header float float_interpolation_trilinear( float v[8], float x, float y, float z )
{
	float	a = ( 1.0f - x ), b = ( 1.0f - y ), c = ( 1.0f - z );
	return ( v[0]*a*b*c + v[1]*x*b*c + v[2]*a*y*c + v[3]*a*b*z + v[4]*x*b*z + v[5]*a*y*z + v[6]*x*y*c + v[7]*x*y*z );
}

#if defined(__cplusplus)
}
#endif

#endif /* !_CU_MATH_FUNCTIONS_H */

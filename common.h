/**
	\file		common.h
	\brief		Common definitions
	\author		Seongoh Lee
	\version	1.0
	\date		2010.09.08
*/

#ifndef _NW_COMMON_H
#define _NW_COMMON_H

//#include "new_well/config.h"
//¿œΩ√¿˚
#include <string.h>

#ifdef SINGLE_PRECISION
typedef float PRECISION; ///< default precision
#endif
#ifdef DOUBLE_PRECISION
typedef double PRECISION; ///< default precision
#endif

#ifndef BOOL
#ifdef NW_OS_APPLE
typedef signed char BOOL; ///< boolean (1 bytes)
#else
typedef int BOOL; ///< boolean (4 bytes)
#endif
#endif

#ifndef BYTE
typedef unsigned char BYTE; ///< unsigned character (1 byte)
#endif
#ifndef UINT
typedef unsigned int UINT; ///< unsigned integer (4 bytes)
#endif
#ifndef WORD
typedef unsigned short WORD; ///< unsigned short integer (2 bytes)
#endif
#ifndef DWORD
typedef unsigned long DWORD; ///< unsigned long (4 bytes)
#endif
#ifndef NULL
#define NULL	((void *)0) ///< null (0)
#endif
#ifndef INT64
typedef __int64 INT64;
#endif
#ifndef UINT64
typedef unsigned __int64 UINT64;
#endif
#ifndef SUCCESS
#define	SUCCESS	0 ///< most useful return value (success)
#endif
#ifndef	FAIL
#define FAIL	1 ///< most useful return value (fail)
#endif
#ifndef TRUE
#define TRUE	1 ///< true
#endif
#ifndef FALSE
#define FALSE	0 ///< false
#endif


/****************************************************************************************\
*						Type conversion with boundary consideration		                 *
\****************************************************************************************/

#define nwf_zero(v)		((v)<FLT_EPSILON && (v)>-FLT_EPSILON) ///< Is v zero?

#define nwf_non_zero(v) ((v)>FLT_EPSILON || (v)<-FLT_EPSILON) ///< Is v non-zero?
#define nwd_non_zero(v) ((v)>DBL_EPSILON || (v)<-DBL_EPSILON) ///< Is v non-zero?

#define nw_double_to_float(d,f)	{ if((d)>3.402823466e+38) (f)=FLT_MAX; else if((d)<-3.402823466e+38) (f)=-FLT_MAX; \
									else if((d)<1.175494351e-38 && (d)>0.0) (f)=FLT_MIN; \
									else if((d)>-1.175494351e-38 && (d)<0.0) (f)=-FLT_MIN; else (f)=(float)(d); }

/****************************************************************************************\
*								Logical definitions						                 *
\****************************************************************************************/
#define nw_swap(a,b,t)	{ (t)=(a);(a)=(b);(b)=(t); } ///< general swap operation (all types)

/****************************************************************************************\
*								Image definitions						                 *
\****************************************************************************************/
#ifdef IP_ALIGNED_FORMAT
// 32-bit aligned format (e.g. IplImage) 
#define IMG_WIDTH_STEP(s,w,c) {int t;(s)=(w)*(c);t=(s)%4;if(t!=0)(s)+=(4-t);} ///< get image line size (size(s),width(w),channels(c))
#else
#define IMG_WIDTH_STEP(s,w,c) ((s)=(w)*(c)) ///< get image line size (size(s),width(w),channels(c))
#endif

/****************************************************************************************\
*								Mathematical definitions					             *
\****************************************************************************************/
#ifndef M_PI		
#define M_PI       3.14159265358979323846 ///< pi
#endif

#ifndef M_SQRT1_2	
#define M_SQRT1_2  0.707106781186547524401 ///< 1/sqrt(2)
#endif

#define M_D2R	0.0174532925199432 ///< PI/180

#define M_SQR(a) ((a)*(a)) ///< square of a number

#define nw_degree_to_radian(a)		((a)*M_D2R) ///< degree to radian

// memory operations
#define alloc_x(x,dim)		(x*)malloc(sizeof(x)*(dim)) ///< memory allocation
#define alloc_char(dim)		(char*)malloc(dim) ///< memory allocation
#define alloc_uint(dim)		alloc_x(UINT,dim) ///< memory allocation
#define alloc_int(dim)		alloc_x(int,dim) ///< memory allocation
#define alloc_float(dim)	alloc_x(float,dim) ///< memory allocation
#define alloc_double(dim)	alloc_x(double,dim) ///< memory allocation

#define calloc_x(x,dim)		(x*)calloc(sizeof(x),dim) ///< Allocate space for array in memory and initializes all its bits to zero.
#define calloc_byte(dim)	(BYTE*)calloc(1,dim) ///< Allocate space for array in memory and initializes all its bits to zero.
#define calloc_int(dim)		calloc_x(int,dim) ///< Allocate space for array in memory and initializes all its bits to zero.


#define copy_x(x,dst,src,dim)		memcpy(dst,src,sizeof(x)*(dim)) ///< memory copy

#define copy_char(dst,src,dim)		memcpy(dst,src,dim) ///< memory copy
#define copy_float(dst,src,dim)		copy_x(float,dst,src,dim) ///< memory copy

#define zeros_x(x,a,dim)	memset(a,0,sizeof(x)*(dim)) ///< fill zeros
#define zeros_float(a,dim)	zeros_x(float,a,dim) ///< fill zeros

#endif // _NW_COMMON_H

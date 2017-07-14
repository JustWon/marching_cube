/**
	\file		con_ui.c
	\brief		Console user interface library source
	\author		Seong-Oh Lee
	\version	1.0
	\date		2009.08.17
*/

#include "con_ui.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include <time.h>
//#include "new_well/config.h"

#ifdef CUI_LOGGING
#define CUI_LOGFILE	"logging.txt" ///< log file name
#endif

// change text color in Windows console mode
// colors are 0=black 1=blue 2=green and so on to 15=white
// colorattribute = foreground + background * 16
// to get red text on yellow use 4 + 14*16 = 228
// light red on yellow would be 12 + 14*16 = 236
// tested with Pelles C  (vegaseat)

#if defined(__cplusplus)
extern "C" {
#endif
void exactinit();
#if defined(__cplusplus)
}
#endif

/// console UI initialization (GREEN text, etc.)

/// \remarks We don't need terminate routine.
void nw_cui_initialize()
{
#ifdef NW_OS_WIN32	
	HWND	console_wnd = GetConsoleWindow();
	HANDLE	console_handle = GetStdHandle(STD_OUTPUT_HANDLE);
	COORD	buffer_size;
	SMALL_RECT	window_size;

	//SetConsoleWindowInfo( console_handle, TRUE, &window_size );
	// Green text
	SetConsoleTextAttribute( console_handle, FOREGROUND_GREEN | FOREGROUND_INTENSITY );
	// Window position	
	MoveWindow( console_wnd, 0, 0, 768, 1024, TRUE );
#elif defined NW_OS_APPLE
#endif
	// Seed the random-number generator with the current time so that
	// the numbers will be different every time we run.
	srand( (unsigned)time( NULL ) );
	exactinit();
}

/// print to console error message

/// \param format	[in] error message
void nw_cui_error( const char *format, ... )
{
#ifdef CUI_ERROR
	char	str[_MAX_PATH];
#if defined NW_OS_WIN32
	HANDLE	console_handle = GetStdHandle(STD_OUTPUT_HANDLE);	
	{
		va_list arg;
		va_start(arg, format);
		vsprintf( str, format, arg );
		va_end(arg);
	}
	// Red text
	SetConsoleTextAttribute( console_handle, FOREGROUND_RED | FOREGROUND_INTENSITY );
	printf( "ERR:	%s\n", str );
	// Green text
	SetConsoleTextAttribute( console_handle, FOREGROUND_GREEN | FOREGROUND_INTENSITY );
#elif defined NW_OS_APPLE
	{
		va_list arg;
		va_start(arg, format);
		vsprintf( str, format, arg );
		va_end(arg);
	}
	printf( "ERR:   %s\n", str );
#endif
#ifdef CUI_LOGGING
	{
		FILE	*fp = fopen( CUI_LOGFILE, "a+" );
		if( fp )
		{
			fprintf( fp, "ERR:   %s\n", str );
			fclose( fp );
		}
	}
#endif
#endif // CUI_ERROR
}

/// print to console normal message

/// \param format	[in] normal message
void nw_cui_message( const char *format, ... )
{
#if defined CUI_MESSAGE
	char	str[_MAX_PATH];
	{
		va_list arg;
		va_start(arg, format);
		vsprintf( str, format, arg );
		va_end(arg);
	}
	printf( "MSG:	%s\n", str );
#ifdef CUI_LOGGING
	{
		FILE	*fp = fopen( CUI_LOGFILE, "a+" );
		if( fp )
		{
			fprintf( fp, "MSG:   %s\n", str );
			fclose( fp );
		}
	}
#endif
#endif // CUI_MESSAGE
}

/// print to console normal message

/// satart.
/// \param format	[in] normal message
void nw_cui_message_s( const char *format, ... )
{
#if defined CUI_MESSAGE	
	char	str[_MAX_PATH];
	va_list arg;
	va_start(arg, format);
	vsprintf( str, format, arg );
	va_end(arg);
	printf( "MSG:	%s", str );
#endif
}

/// print to console normal message

/// end.
/// \param format	[in] normal message
void nw_cui_message_e( const char *format, ... )
{
#if defined CUI_MESSAGE
	char	str[_MAX_PATH];
	va_list arg;
	va_start(arg, format);
	vsprintf( str, format, arg );
	va_end(arg);
	printf( "%s\n", str );
#endif
}

/// print to console debug message

/// \param format	[in] debug message
void nw_cui_debug( const char *format, ... )
{
#if defined CUI_DEBUG
	char	str[_MAX_PATH];
#if defined NW_OS_WIN32	
	HANDLE	console_handle = GetStdHandle(STD_OUTPUT_HANDLE);
	{
		va_list arg;
		va_start(arg, format);
		vsprintf( str, format, arg );
		va_end(arg);
	}
	// Red text
	SetConsoleTextAttribute( console_handle, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY );
	printf( "DEBUG:	%s\n", str );
	// Green text
	SetConsoleTextAttribute( console_handle, FOREGROUND_GREEN | FOREGROUND_INTENSITY );
#elif defined NW_OS_APPLE
	{
		va_list arg;
		va_start(arg, format);
		vsprintf( str, format, arg );
		va_end(arg);
	}
	printf( "DEBUG: %s\n", str );
#endif
#ifdef CUI_LOGGING
	{
		FILE	*fp = fopen( CUI_LOGFILE, "a+" );
		if( fp )
		{
			fprintf( fp, "DEBUG: %s\n", str );
			fclose( fp );
		}
	}
#endif
#endif // CUI_DEBUG
}

/**
	\file		con_ui.h
	\brief		Console user interface library header
	\author		Seong-Oh Lee
	\version	1.0
	\date		2009.08.17
*/

#ifndef _NW_CONSOLE_UI_H
#define _NW_CONSOLE_UI_H

#include "common.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define CUI_ERROR ///< activate error message
#define CUI_WARNING ///< activate warning message
#define CUI_MESSAGE ///< activate normal message
#define CUI_DEBUG ///< activate debug message
#define CUI_LOGGING ///< enable message logging

void nw_cui_initialize();

void nw_cui_error( const char *format, ... );
void nw_cui_message( const char *format, ... );
void nw_cui_message_s( const char *format, ... );
void nw_cui_message_e( const char *format, ... );
void nw_cui_debug( const char *format, ... );

#if defined(__cplusplus)
}
#endif

#endif

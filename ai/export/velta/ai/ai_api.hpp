#pragma once

#ifdef AI_LIBRARY
#define AI_API EXPORT_MACRO
#else
#define AI_API IMPORT_MACRO
#endif
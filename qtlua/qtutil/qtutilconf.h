// -*- C -*-

#ifndef QTUTILCONF_H
#define QTUTILCONF_H

#ifdef WIN32
# ifdef libqtutil_EXPORTS
#  define QTUTILAPI __declspec(dllexport)
# else
#  define QTUTILAPI __declspec(dllimport)
# endif
#else
# define QTUTILAPI
#endif

#ifdef __cplusplus
# define QTUTIL_EXTERNC extern "C"
#else
# define QTUTIL_EXTERNC extern
#endif

#endif




/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "lua_[A-Z]\\sw*[a-z]\\sw*" )
   c-font-lock-extra-types: ( "\\sw+_t" "lua_[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */

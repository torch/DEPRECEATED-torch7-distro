/* -*- C -*- */


#include "paths.h"




/* ------------------------------------------------------ */
/* Utils to manipulate strings */


#define SBINCREMENT 256

typedef struct {
  char *buffer;
  int maxlen;
  int len;
} SB;

static void 
sbinit(SB *sb)
{
  sb->buffer = (char*)malloc(SBINCREMENT);
  sb->maxlen = SBINCREMENT;
  sb->len = 0;
}

static char *
sbfree(SB *sb)
{
  if (sb->buffer)
    free(sb->buffer);
  sb->buffer = 0;
  return 0;
}

static void
sbgrow(SB *sb, int n)
{
  if (sb->buffer && sb->len + n > sb->maxlen)
    {
      int nlen = sb->maxlen;
      while (sb->len + n > nlen)
        nlen += SBINCREMENT;
      sb->buffer = (char*)realloc(sb->buffer, nlen);
      sb->maxlen = nlen;
    }
}

static void
sbadd1(SB *sb, char c)
{
  sbgrow(sb, 1);
  if (sb->buffer)
    sb->buffer[sb->len++] = c;
}

static void
sbaddn(SB *sb, const char *s, int n)
{
  sbgrow(sb, n);
  if (sb->buffer && s && n)
    memcpy(sb->buffer + sb->len, s, n);
  else if (sb->buffer && n)
    sbfree(sb);
  sb->len += n;
}

static void
sbaddsf(SB *sb, char *s)
{
  if (s)
    sbaddn(sb, s, strlen(s));
  else
    sbfree(sb);
  if (s)
    free((void*)s);
}

static void
sbslash(SB *sb)
{
  int i;
  if (sb->buffer && sb->len)
    for(i=0; i<sb->len; i++)
      if (sb->buffer[i]=='\\')
        sb->buffer[i]='/';
}

static int
sbpush(lua_State *L, SB *sb)
{
  sbslash(sb);
  lua_pushlstring(L, sb->buffer, sb->len);
  sbfree(sb);
  return 1;
}

static int
sbsetpush(lua_State *L,  SB *sb, const char *s)
{
  sbfree(sb);
  lua_pushstring(L, s);
  return 1;
}


/* ------------------------------------------------------ */
/* filep, dirp, basename, dirname */


static int 
filep(lua_State *L)
{
  const char *s = luaL_checkstring(L, 1);
#ifdef LUA_WIN
  struct _stat buf;
  if (_stat(s,&buf) < 0)
    return 0;
  if (buf.st_mode & S_IFDIR) 
    return 0;
#else
  struct stat buf;
  if (stat(s,&buf) < 0)
    return 0;
  if (buf.st_mode & S_IFDIR) 
    return 0;
#endif
  return 1;
}


static int 
dirp(lua_State *L)
{
  const char *s = luaL_checkstring(L, 1);
#ifdef LUA_WIN
  struct _stat buf;
  const char *last;
  if ((s[0]=='/' || s[0]=='\\') && 
      (s[1]=='/' || s[1]=='\\') && !s[2]) 
    return 1;
  last = s + strlen(s);
  if (*last=='/' || *last=='\\' || *last==':')
    {
      lua_settop(L, 1);
      lua_pushliteral(L, ".");
      lua_concat(L, 2);
      s = lua_tostring(L, -1);
    }
  if (_stat(s,&buf)==0)
    if (buf.st_mode & S_IFDIR)
      return 1;
#else
  struct stat buf;
  if (stat(s,&buf)==0)
    if (buf.st_mode & S_IFDIR)
      return 1;
#endif
  return 0;
}


static int
lua_filep(lua_State *L)
{
  lua_pushboolean(L, filep(L));
  return 1;
}


static int
lua_dirp(lua_State *L)
{
  lua_pushboolean(L, dirp(L));
  return 1;
}


static int 
lua_basename(lua_State *L)
{
  const char *fname = luaL_checkstring(L, 1);
  const char *suffix = luaL_optstring(L, 2, 0);

#ifdef LUA_WIN

  int sl;
  const char *p, *s;
  SB sb;
  sbinit(&sb);
  /* Special cases */
  if (fname[0] && fname[1]==':') {
    sbaddn(&sb, fname, 2);
    fname += 2;
    if (fname[0]=='/' || fname[0]=='\\')
      sbadd1(&sb, '/');
    while (fname[0]=='/' || fname[0]=='\\')
      fname += 1;
    if (fname[0]==0)
      return sbpush(L, &sb);
    sb.len = 0;
  }
  /* Position p after last nontrivial slash */
  s = p = fname;
  while (*s) {
    if ((s[0]=='\\' || s[0]=='/') &&
        (s[1] && s[1]!='/' && s[1]!='\\' ) )
      p = s + 1;
    s++;
  }
  /* Copy into buffer */
  while (*p && *p!='/' && *p!='\\')
    sbadd1(&sb, *p++);
  /* Process suffix */
  if (suffix==0 || suffix[0]==0)
    return sbpush(L, &sb);
  if (suffix[0]=='.')
    suffix += 1;
  if (suffix[0]==0)
    return sbpush(L, &sb);
  sl = strlen(suffix);
  if (sb.len > sl) {
    s =  sb.buffer + sb.len - (sl + 1);
    if (s[0]=='.' && _strnicmp(s+1,suffix, sl)==0)
      sb.len = s - sb.buffer;
  }
  return sbpush(L, &sb);
  
#else

  int sl;
  const char *s, *p;
  SB sb;
  sbinit(&sb);
  /* Position p after last nontrivial slash */
  s = p = fname;
  while (*s) {
    if (s[0]=='/' && s[1] && s[1]!='/')
      p = s + 1;
    s++;
  }
  /* Copy into buffer */
  while (*p && *p!='/')
    sbadd1(&sb, *p++);
  /* Process suffix */
  if (suffix==0 || suffix[0]==0)
    return sbpush(L, &sb);
  if (suffix[0]=='.')
    suffix += 1;
  if (suffix[0]==0)
    return sbpush(L, &sb);
  sl = strlen(suffix);
  if (sb.len > sl) {
    s =  sb.buffer + sb.len - (sl + 1);
    if (s[0]=='.' && strncmp(s+1,suffix, sl)==0)
      sb.len = s - sb.buffer;
  }
  return sbpush(L, &sb);

#endif
}


static int 
lua_dirname(lua_State *L)
{
  const char *fname = luaL_checkstring(L, 1);

#ifdef LUA_WIN

  const char *s;
  const char *p;
  SB sb;
  sbinit(&sb);
  /* Handle leading drive specifier */
  if (isalpha((unsigned char)fname[0]) && fname[1]==':') {
    sbadd1(&sb, *fname++);
    sbadd1(&sb, *fname++);
  }
  /* Search last non terminal / or \ */
  p = 0;
  s = fname;
  while (*s) {
    if ((s[0]=='\\' || s[0]=='/') &&
        (s[1] && s[1]!='/' && s[1]!='\\') )
      p = s;
    s++;
  }
  /* Cannot find non terminal / or \ */
  if (p == 0) {
    if (sb.len > 0) {
      if (fname[0]==0 || fname[0]=='/' || fname[0]=='\\')
	sbadd1(&sb, '/');
      return sbpush(L, &sb);
    } else {
      if (fname[0]=='/' || fname[0]=='\\')
	return sbsetpush(L, &sb, "//");
      else
	return sbsetpush(L, &sb, ".");
    }
  }
  /* Single leading slash */
  if (p == fname) {
    sbadd1(&sb, '/');
    return sbpush(L, &sb);
  }
  /* Backtrack all slashes */
  while (p>fname && (p[-1]=='/' || p[-1]=='\\'))
    p--;
  /* Multiple leading slashes */
  if (p == fname)
    return sbsetpush(L, &sb, "//");
  /* Regular case */
  s = fname;
  do {
    sbadd1(&sb, *s++);
  } while (s<p);
  return sbpush(L, &sb);

#else

  const char *s = fname;
  const char *p = 0;
  SB sb; 
  sbinit(&sb);
  while (*s) {
    if (s[0]=='/' && s[1] && s[1]!='/')
      p = s;
    s++;
  }
  if (!p) {
    if (fname[0]=='/')
      return sbsetpush(L, &sb, fname);
    else
      return sbsetpush(L, &sb, ".");
  }
  s = fname;
  do {
    sbadd1(&sb, *s++);
  } while (s<p);
  return sbpush(L, &sb);

#endif
}



/* ------------------------------------------------------ */
/* cwd and concat */


static int
lua_cwd(lua_State *L)
{
#ifdef LUA_WIN

  char drv[2];
  int l;
  SB sb;
  sbinit(&sb);
  drv[0] = '.'; drv[1] = 0;
  l = GetFullPathNameA(drv, sb.maxlen, sb.buffer, 0);
  if (l > sb.maxlen) {
    sbgrow(&sb, l+1);
    l = GetFullPathNameA(drv, sb.maxlen, sb.buffer, 0);
  }
  if (l <= 0)
    return sbsetpush(L, &sb, ".");
  sb.len += l;
  return sbpush(L, &sb);

#elif HAVE_GETCWD
  
  const char *s;
  SB sb;
  sbinit(&sb);
  s = getcwd(sb.buffer, sb.maxlen);
  while (!s && errno==ERANGE)
    {
      sbgrow(&sb, sb.maxlen + SBINCREMENT);
      s = getcwd(sb.buffer, sb.maxlen);
    }
  if (! s)
    return sbsetpush(L, &sb, ".");
  sb.len += strlen(s);
  return sbpush(L, &sb);

#else
  
  const char *s;
  SB sb;
  sbinit(&sb);
  sbgrow(&sb, PATH_MAX); 
  s = getwd(sb.buffer);
  if (! s)
    return sbsetpush(L, &sb, ".");
  sb.len += strlen(s);
  return sbpush(L, &sb);

#endif
}



static int 
concat_fname(lua_State *L, const char *fname)
{
  const char *from = lua_tostring(L, -1);

#ifdef LUA_WIN

  const char *s;
  SB sb;
  sbinit(&sb);
  sbaddn(&sb, from, strlen(from));
  if (fname==0)
    return sbpush(L, &sb);
  /* Handle absolute part of fname */
  if (fname[0]=='/' || fname[0]=='\\') {
    if (fname[1]=='/' || fname[1]=='\\') {
      sb.len = 0;                            /* Case //abcd */
      sbaddn(&sb, "//", 2);
    } else {
      char drive;
      if (sb.len >= 2 && sb.buffer[1]==':'   /* Case "/abcd" */
          && isalpha((unsigned char)(sb.buffer[0])) )
        drive = sb.buffer[0];
      else
        drive = _getdrive() + 'A' - 1;
      sb.len = 0;
      sbadd1(&sb, drive);
      sbaddn(&sb, ":/", 2);
    }
  } else if (fname[0] && 	              /* Case "x:abcd"   */
             isalpha((unsigned char)(fname[0])) && fname[1]==':') {
    if (fname[2]!='/' && fname[2]!='\\') {
      if (sb.len < 2 || sb.buffer[1]!=':' 
          || !isalpha((unsigned char)(sb.buffer[0]))
          || (toupper((unsigned char)sb.buffer[0]) !=
              toupper((unsigned char)fname[0]) ) ) 
        {
          int l;
          char drv[4];
          sb.len = 0;
          drv[0]=fname[0]; drv[1]=':'; drv[2]='.'; drv[3]=0;
          l = GetFullPathNameA(drv, sb.maxlen, sb.buffer, 0);
          if (l > sb.maxlen) {
            sbgrow(&sb, l+1);
            l = GetFullPathNameA(drv, sb.maxlen, sb.buffer, 0);
          }
          if (l <= 0)
            sbaddn(&sb, drv, 3);
          else
            sb.len += l;
        }
      fname += 2;
    } else {
      sb.len = 0;                              /* Case "x:/abcd"  */
      sbadd1(&sb, toupper((unsigned char)fname[0])); 
      sbaddn(&sb, ":/", 2);
      fname += 2;
      while (*fname == '/' || *fname == '\\')
        fname += 1;
    }
  }
  /* Process path components */
  for (;;)
  {
    while (*fname=='/' || *fname=='\\')
      fname ++;
    if (*fname == 0)
      return sbpush(L, &sb);
    if (fname[0]=='.') {
      if (fname[1]=='/' || fname[1]=='\\' || fname[1]==0) {
	fname += 1;
	continue;
      }
      if (fname[1]=='.')
        if (fname[2]=='/' || fname[2]=='\\' || fname[2]==0) {
          size_t l;
	  fname += 2;
          lua_pushcfunction(L, lua_dirname);
          sbpush(L, &sb);
          lua_call(L, 1, 1);
          s = lua_tolstring(L, -1, &l);
          sbinit(&sb);
          sbaddn(&sb, s, l);
          lua_pop(L, 1);
	  continue;
      }
    }
    if (sb.len==0 || 
        (sb.buffer[sb.len-1]!='/' && sb.buffer[sb.len-1]!='\\') )
      sbadd1(&sb, '/');
    while (*fname && *fname!='/' && *fname!='\\')
      sbadd1(&sb, *fname++);
  }
             
#else
  
  const char *s;
  SB sb;
  sbinit(&sb);

  if (fname && fname[0]=='/') 
    sbadd1(&sb, '/');
  else
    sbaddn(&sb, from, strlen(from));
  for (;;) {
    while (fname && fname[0]=='/')
      fname++;
    if (!fname || !fname[0]) {
      sbadd1(&sb, '/');
      while (sb.len > 1 && sb.buffer[sb.len-1]=='/')
        sb.len --;
      return sbpush(L, &sb);
    }
    if (fname[0]=='.') {
      if (fname[1]=='/' || fname[1]==0) {
	fname +=1;
	continue;
      }
      if (fname[1]=='.')
	if (fname[2]=='/' || fname[2]==0) {
	  fname +=2;
          while (sb.len > 0 && sb.buffer[sb.len-1]=='/')
            sb.len --;
          while (sb.len > 0 && sb.buffer[sb.len-1]!='/')
            sb.len --;
	  continue;
	}
    }
    if (sb.len == 0 || sb.buffer[sb.len-1] != '/')
      sbadd1(&sb, '/');
    while (*fname!=0 && *fname!='/')
      sbadd1(&sb, *fname++);
  }
  
  
#endif

}


static int
lua_concatfname(lua_State *L)
{
  int i;
  int narg = lua_gettop(L);
  lua_cwd(L);
  for (i=1; i<=narg; i++)
    {
      concat_fname(L, luaL_checkstring(L, i));
      lua_remove(L, -2);
    }
  return 1;
}



/* ------------------------------------------------------ */
/* execdir */


static int 
lua_execdir(lua_State *L)
{
  const char *s = 0;
#if HAVE_LUA_EXECUTABLE_DIR
  s =  lua_executable_dir(0);
#endif
  if (s && s[0])
    lua_pushstring(L, s);
  else
    lua_pushnil(L);
  return 1;
}



/* ------------------------------------------------------ */
/* file lists */


static int
lua_dir(lua_State *L)
{
  int k = 0;
  const char *s = luaL_checkstring(L, 1);

#ifdef LUA_WIN

  SB sb;
  struct _finddata_t info;
  long hfind;
  /* special cases */
  lua_createtable(L, 0, 0);
  if ((s[0]=='/' || s[0]=='\\') && 
      (s[1]=='/' || s[1]=='\\') && !s[2]) 
    {
      int drive;
      hfind = GetLogicalDrives();
      for (drive='A'; drive<='Z'; drive++)
        if (hfind & (1<<(drive-'A'))) {
          lua_pushfstring(L, "%c:/", drive);
          lua_rawseti(L, -2, ++k);
        }
    } 
  else if (dirp(L)) {
    lua_pushliteral(L, "..");
    lua_rawseti(L, -2, ++k);
  } else {
    lua_pushnil(L);
    return 1;
  }
  /* files */
  sbinit(&sb);
  sbaddn(&sb, s, strlen(s));
  if (sb.len>0 && sb.buffer[sb.len-1]!='/' && sb.buffer[sb.len-1]!='\\')
    sbadd1(&sb, '/');
  sbaddn(&sb, "*.*", 3);
  sbadd1(&sb, 0);
  hfind = _findfirst(sb.buffer, &info);
  if (hfind != -1) {
    do {
      if (strcmp(".",info.name) && strcmp("..",info.name)) {
        lua_pushstring(L, info.name);
        lua_rawseti(L, -2, ++k);
      }
    } while ( _findnext(hfind, &info) != -1 );
    _findclose(hfind);
  }
  sbfree(&sb);

#else

  DIR *dirp;
  struct dirent *d;
  dirp = opendir(s);
  if (dirp) {
    lua_createtable(L, 0, 0);
    while ((d = readdir(dirp))) {
      int n = NAMLEN(d);
      lua_pushlstring(L, d->d_name, n);
      lua_rawseti(L, -2, ++k);
    }
    closedir(dirp);
  } else
    lua_pushnil(L);

#endif
  
  return 1;
}



/* ------------------------------------------------------ */
/* require (with global flag) */

#ifdef LUA_DL_DLOPEN
# define NEED_PATH_REQUIRE 1
# include <dlfcn.h>
# ifndef RTLD_LAZY
#  define RTLD_LAZY 1
# endif
# ifndef RTLD_GLOBAL
#  define RTLD_GLOBAL 0
# endif
# define LL_LOAD(h,fname) h=dlopen(fname,RTLD_LAZY|RTLD_GLOBAL)
# define LL_SYM(h,sym) dlsym(h, sym)
#endif

#ifdef LUA_DL_DLL
# define NEED_PATH_REQUIRE 1
# include <windows.h>
# define LL_LOAD(h,fname) h=(void*)LoadLibraryA(fname)
# define LL_SYM(h,sym) GetProcAddress((HINSTANCE)h,sym)
#endif

#if NEED_PATH_REQUIRE

// {{{ functions copied or derived from loadlib.c

static int readable (const char *filename) 
{  
  FILE *f = fopen(filename, "r");  /* try to open file */
  if (f == NULL) return 0;  /* open failed */
  fclose(f);
  return 1;
}

static const char *pushnexttemplate (lua_State *L, const char *path) 
{
  const char *l;
  while (*path == *LUA_PATHSEP) path++;  /* skip separators */
  if (*path == '\0') return NULL;  /* no more templates */
  l = strchr(path, *LUA_PATHSEP);  /* find next separator */
  if (l == NULL) l = path + strlen(path);
  lua_pushlstring(L, path, l - path);  /* template */
  return l;
}

static const char *pushfilename (lua_State *L, const char *name) 
{
  const char *path;
  const char *filename;
  lua_getfield(L, LUA_GLOBALSINDEX, "package");
  lua_getfield(L, -1, "cpath");
  lua_remove(L, -2);
  if (! (path = lua_tostring(L, -1)))
    luaL_error(L, LUA_QL("package.cpath") " must be a string");
  lua_pushliteral(L, ""); 
  while ((path = pushnexttemplate(L, path))) {
    filename = luaL_gsub(L, lua_tostring(L, -1), "?", name);
    lua_remove(L, -2);
    if (readable(filename))
      { // stack:  cpath errmsg filename
        lua_remove(L, -3);
        lua_remove(L, -2);
        return lua_tostring(L, -1);
      }
    lua_pushfstring(L, "\n\tno file " LUA_QS, filename);
    lua_remove(L, -2); /* remove file name */
    lua_concat(L, 2);  /* add entry to possible error message */
  }
  lua_pushfstring(L, "module " LUA_QS " not found", name);
  lua_replace(L, -3);
  lua_concat(L, 2);
  lua_error(L);
  return 0;
}

// functions copied or derived from loadlib.c }}}

static int
path_require(lua_State *L)
{
  const char *filename;
  lua_CFunction func;
  void *handle;
  const char *name = luaL_checkstring(L, 1);
  lua_settop(L, 1);
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");  // index 2
  lua_getfield(L, 2, name);
  if (lua_toboolean(L, -1))
    return 1;
  filename = pushfilename(L, name);  // index 3
  LL_LOAD(handle, filename);
  if (! handle)
    luaL_error(L, "cannot load " LUA_QS, filename);
  lua_pushfstring(L, "luaopen_%s", name);  // index 4
  func = (lua_CFunction)LL_SYM(handle, lua_tostring(L, -1));
  if (! func)
    luaL_error(L, "no symbol " LUA_QS " in module " LUA_QS, 
               lua_tostring(L, -1), filename);
  lua_pushboolean(L, 1);
  lua_setfield(L, 2, name);
  lua_pushcfunction(L, func);
  lua_pushstring(L, name);
  lua_call(L, 1, 1);
  if (! lua_isnil(L, -1))
    lua_setfield(L, 2, name);
  lua_getfield(L, 2, name);
  return 1;
}

#else

// fallback to calling require

static int
path_require(lua_State *L)
{
  int narg = lua_gettop(L);
  lua_getfield(L, LUA_GLOBALSINDEX, "require");
  lua_insert(L, 1);
  lua_call(L, narg, 1);
  return 1;
}

#endif


/* ------------------------------------------------------ */
/* register */


static const struct luaL_Reg paths__ [] = {
  {"filep", lua_filep},
  {"dirp", lua_dirp},
  {"basename", lua_basename},
  {"dirname", lua_dirname},
  {"cwd", lua_cwd},
  {"concat", lua_concatfname},
  {"execdir", lua_execdir},
  {"dir", lua_dir},
  {"require", path_require},
  {NULL, NULL}
};


PATHS_API int 
luaopen_libpaths(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "paths");
  luaL_register(L, NULL, paths__);
  return 1;
}

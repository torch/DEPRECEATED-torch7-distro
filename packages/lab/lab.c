#include "TH.h"
#include "luaT.h"
#include "utils.h"

#include "sys/time.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)

#define lab_(NAME) TH_CONCAT_3(lab_, Real, NAME)

static const void* torch_ByteTensor_id;
static const void* torch_CharTensor_id;
static const void* torch_ShortTensor_id;
static const void* torch_IntTensor_id;
static const void* torch_LongTensor_id;
static const void* torch_FloatTensor_id;
static const void* torch_DoubleTensor_id;

static const void* torch_LongStorage_id;

static const void* lab_default_tensor_id;

#include "labwrap.c"

/* #include "generic/lab.c" */
/* #include "THGenerateAllTypes.h" */

#include "generic/labconv.c"
#include "THGenerateAllTypes.h"

#include "generic/lablapack.c"
#include "THGenerateFloatTypes.h"

static int lab_setdefaulttensortype(lua_State *L)
{
  const void *id;

  luaL_checkstring(L, 1);
  
  if(!(id = luaT_typename2id(L, lua_tostring(L, 1))))                  \
    return luaL_error(L, "<%s> is not a string describing a torch object", lua_tostring(L, 1)); \

  lab_default_tensor_id = id;

  return 0;
}

static int lab_getdefaulttensortype(lua_State *L)
{
  lua_pushstring(L, luaT_id2typename(L, lab_default_tensor_id));
  return 1;
}

static int lab_tic(lua_State* L)
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  double ttime = (double)tv.tv_sec + (double)(tv.tv_usec)/1000000.0;
  lua_pushnumber(L,ttime);
  return 1;
}

static int lab_toc(lua_State* L)
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  double toctime = (double)tv.tv_sec + (double)(tv.tv_usec)/1000000.0;
  lua_Number tictime = luaL_checknumber(L,1);
  lua_pushnumber(L,toctime-tictime);
  return 1;
}

static const struct luaL_Reg lab_basics [] = {
  {"setdefaulttensortype", lab_setdefaulttensortype},
  {"getdefaulttensortype", lab_getdefaulttensortype},
  {"tic", lab_tic},
  {"toc", lab_toc},
  {NULL, NULL}
};

void lab_init(lua_State *L)
{
  lab_ByteTensor_init(L);
  lab_CharTensor_init(L);
  lab_ShortTensor_init(L);
  lab_IntTensor_init(L);
  lab_LongTensor_init(L);
  lab_FloatTensor_init(L);
  lab_DoubleTensor_init(L);

  luaL_register(L, NULL, lab_basics);
  luaL_register(L, NULL, lab_stuff);

  lab_default_tensor_id = torch_DoubleTensor_id; /* obviously, this must be done after the "generic" stuff */
}

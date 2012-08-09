// -*- C++ -*-

#include "qttorch.h"

#include <QColor>
#include <QDebug>
#include <QImage>
#include <QMetaType>

#include "TH.h"
#include "luaT.h"

static const char* torch_ByteTensor_id;
static const char* torch_CharTensor_id;
static const char* torch_ShortTensor_id;
static const char* torch_IntTensor_id;
static const char* torch_LongTensor_id;
static const char* torch_FloatTensor_id;
static const char* torch_DoubleTensor_id;

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define qttorch_(NAME) TH_CONCAT_3(qttorch_, Real, NAME)

#include "generic/qttorch.cpp"
#include "THGenerateAllTypes.h"

struct luaL_Reg qttorch_qimage_lib[] = {
  {"toByteTensor", qttorch_Byteqimage_totensor},
  {"toCharTensor", qttorch_Charqimage_totensor},
  {"toShortTensor", qttorch_Shortqimage_totensor},
  {"toIntTensor", qttorch_Intqimage_totensor},
  {"toLongTensor", qttorch_Longqimage_totensor},
  {"toFloatTensor", qttorch_Floatqimage_totensor},
  {"toDoubleTensor", qttorch_Doubleqimage_totensor},
  {0,0}
};

int luaopen_libqttorch(lua_State *L)
{
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);
  // load modules 'torch'
  if (luaL_dostring(L, "require 'torch'"))
    lua_error(L);

  torch_ByteTensor_id = luaT_checktypename2id(L, "torch.ByteTensor");
  torch_CharTensor_id = luaT_checktypename2id(L, "torch.CharTensor");
  torch_ShortTensor_id = luaT_checktypename2id(L, "torch.ShortTensor");
  torch_IntTensor_id = luaT_checktypename2id(L, "torch.IntTensor");
  torch_LongTensor_id = luaT_checktypename2id(L, "torch.LongTensor");
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  qttorch_ByteTensor_init(L);
  qttorch_CharTensor_init(L);
  qttorch_ShortTensor_init(L);
  qttorch_IntTensor_init(L);
  qttorch_LongTensor_init(L);
  qttorch_FloatTensor_init(L);
  qttorch_DoubleTensor_init(L);

  // enrichs QImage
  luaQ_pushmeta(L, QMetaType::QImage);
  luaQ_getfield(L, -1, "__metatable");
  luaL_register(L, 0, qttorch_qimage_lib);
  
  return 0;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ("\\sw+_t" "\\(lua_\\)?[A-Z]\\sw*[a-z]\\sw*")
   End:
   ------------------------------------------------------------- */


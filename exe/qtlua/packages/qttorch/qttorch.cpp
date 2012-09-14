// -*- C++ -*-

#include "qttorch.h"

#include <QColor>
#include <QDebug>
#include <QImage>
#include <QMetaType>

#include "TH.h"
#include "luaT.h"

#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
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


// -*- C++ -*-


#include "qttorch.h"

#include <QColor>
#include <QDebug>
#include <QImage>
#include <QMetaType>

#include "TH.h"
#include "luaT.h"

static const void* torch_Tensor_id;

static int
qttorch_qimage_fromtensor(lua_State *L)
{
  THTensor *Tsrc = (THTensor*)luaT_checkudata(L,1,torch_Tensor_id);
  long depth = 1;
  if ( Tsrc->nDimension == 3)
    depth = Tsrc->size[2];
  else if (Tsrc->nDimension != 2)
    luaL_error(L, "tensor must have 2 or 3 dimensions");
  if (depth != 1 && depth != 3 && depth != 4)
    luaL_error(L, "tensor third dimension must be 1, 3, or 4.");
  // create image
  if (Tsrc->size[0] >= INT_MAX || Tsrc->size[1] >= INT_MAX)
    luaL_error(L, "image is too large");
  int width = (int)(Tsrc->size[0]);
  int height = (int)(Tsrc->size[1]);
  QImage image(width, height, QImage::Format_ARGB32_Premultiplied);
  // fill image
  long s0 = Tsrc->stride[0];
  long s1 = Tsrc->stride[1];
  long s2 = (depth > 1) ? Tsrc->stride[2] : 0;
  double *tdata = THTensor_dataPtr(Tsrc);
  for(int j=0; j<height; j++) 
    {
      QRgb *ip = (QRgb*)image.scanLine(j);
      double *tp = tdata + s1 * j;
      if (depth == 1)
        {
          for (int i=0; i<width; i++)
            {
              int g = (int)(tp[0] * 255.0) & 0xff;
              tp += s0;
              ip[i] = qRgb(g,g,g);
            }
        }
      else if (depth == 3)
        {
          for (int i=0; i<width; i++)
            {
              int r = (int)(tp[0] * 255.0) & 0xff;
              int g = (int)(tp[s2] * 255.0) & 0xff;
              int b = (int)(tp[s2+s2] * 255.0) & 0xff;
              tp += s0;
              ip[i] = qRgb(r,g,b);
            }
        }
      else if (depth == 4)
        {
          for (int i=0; i<width; i++)
            {
              int a = (int)(tp[s2+s2+s2] * 255.0) & 0xff;
              int r = (int)(tp[0] * a) & 0xff;
              int g = (int)(tp[s2] * a) & 0xff;
              int b = (int)(tp[s2+s2] * a) & 0xff;
              tp += s0;
              ip[i] = qRgba(r,g,b,a);
            }
        }
    }
  // return
  luaQ_pushqt(L, image);
  return 1;
}


static int
qttorch_qimage_totensor(lua_State *L)
{
  THTensor *Tdst = 0;
  QImage image = luaQ_checkqvariant<QImage>(L, 1);
  int width = image.width();
  int height = image.height();
  int depth = 1;
  int tpos = 0;
  // validate arguments
  if (lua_type(L, 2) == LUA_TUSERDATA)
    {
      tpos = 2;
      Tdst = (THTensor*)luaT_checkudata(L,2,torch_Tensor_id);
      if (Tdst->nDimension == 3)
        depth = Tdst->size[2];
      else if (Tdst->nDimension != 2)
        luaL_error(L, "tensor must have 2 or 3 dimensions");
      if (depth != 1 && depth != 3 && depth != 4)
        luaL_error(L, "tensor third dimension must be 1, 3, or 4.");
      if (width != Tdst->size[0] || height != Tdst->size[1])
        luaL_error(L, "tensor dimensions must match the image size.");
    }
  else
    {
      depth = luaL_optinteger(L, 2, 3);
      if (depth != 1 && depth != 3 && depth != 4)
        luaL_error(L, "depth must be 1, 3, or 4.");
      if (depth == 1)
        Tdst = THTensor_newWithSize2d(width, height);
      else
        Tdst = THTensor_newWithSize3d(width, height, depth);
    }
  // convert image
  if (image.format() != QImage::Format_ARGB32)
    image = image.convertToFormat(QImage::Format_ARGB32);
  if (image.format() != QImage::Format_ARGB32)
    luaL_error(L, "Cannot convert image to format ARGB32");
  // fill tensor
  long s0 = Tdst->stride[0];
  long s1 = Tdst->stride[1];
  long s2 = (depth > 1) ? Tdst->stride[2] : 0;
  double *tdata = THTensor_dataPtr(Tdst);
  for(int j=0; j<height; j++) 
    {
      QRgb *ip = (QRgb*)image.scanLine(j);
      double *tp = tdata + s1 * j;
      if (depth == 1)
        {
          for (int i=0; i<width; i++)
            {
              QRgb v = ip[i];
              tp[0] = (qreal)qGray(v) / 255.0;
              tp += s0;
            }
        }
      else if (depth == 3)
        {
          for (int i=0; i<width; i++)
            {
              QRgb v = ip[i];
              tp[0] = (qreal)qRed(v) / 255.0;
              tp[s2] = (qreal)qGreen(v) / 255.0;
              tp[s2+s2] = (qreal)qBlue(v) / 255.0;
              tp += s0;
            }
        }
      else if (depth == 4)
        {
          for (int i=0; i<width; i++)
            {
              QRgb v = ip[i];
              tp[0] = (qreal)qRed(v) / 255.0;
              tp[s2] = (qreal)qGreen(v) / 255.0;
              tp[s2+s2] = (qreal)qBlue(v) / 255.0;
              tp[s2+s2+s2] = (qreal)qAlpha(v) / 255.0;
              tp += s0;
            }
        }
    }
  // return
  if (tpos > 0)
    lua_pushvalue(L, tpos);
  else
    luaT_pushudata(L, (void*)Tdst, torch_Tensor_id);
  return 1;
}




struct luaL_Reg qttorch_qimage_lib[] = {
  {"fromTensor", qttorch_qimage_fromtensor},
  {"toTensor", qttorch_qimage_totensor},
  {0,0}
};


int 
luaopen_libqttorch(lua_State *L)
{
  // load module 'qt'
  if (luaL_dostring(L, "require 'qt'"))
    lua_error(L);
  // load modules 'torch'
  if (luaL_dostring(L, "require 'torch'"))
    lua_error(L);
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");

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


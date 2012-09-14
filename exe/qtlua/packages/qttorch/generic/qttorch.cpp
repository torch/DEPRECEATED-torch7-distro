#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/qttorch.cpp"
#else

static int qttorch_(qimage_fromtensor)(lua_State *L)
{
  THTensor *Tsrc = (THTensor*)luaT_checkudata(L,1,torch_Tensor);
  double scale = 255.0 / luaL_optnumber(L, 2, 1);
  long depth = 1;
  int ddim, wdim = 1, hdim = 0;

  if ( Tsrc->nDimension == 3)
  {
    ddim = 0;
    hdim = 1;
    wdim = 2;
    depth = Tsrc->size[ddim];
  }
  else if (Tsrc->nDimension != 2)
    luaL_error(L, "tensor must have 2 or 3 dimensions");
  if (depth != 1 && depth != 3 && depth != 4)
    luaL_error(L, "tensor first dimension must be 1, 3, or 4.");
  // create image
  if (Tsrc->size[wdim] >= INT_MAX || Tsrc->size[hdim] >= INT_MAX)
    luaL_error(L, "image is too large");
  int width = (int)(Tsrc->size[wdim]);
  int height = (int)(Tsrc->size[hdim]);
  QImage image(width, height, QImage::Format_ARGB32_Premultiplied);
  // fill image
  long sw = Tsrc->stride[wdim];
  long sh = Tsrc->stride[hdim];
  long sd = (depth > 1) ? Tsrc->stride[ddim] : 0;
  real *tdata = THTensor_(data)(Tsrc);
  for(int j=0; j<height; j++)
  {
    QRgb *ip = (QRgb*)image.scanLine(j);
    real *tp = tdata + sh * j;
    if (depth == 1)
    {
      for (int i=0; i<width; i++)
      {
        int g = (int)(tp[0] * scale) & 0xff;
        tp += sw;
        ip[i] = qRgb(g,g,g);
      }
    }
    else if (depth == 3)
    {
      for (int i=0; i<width; i++)
      {
        int r = (int)(tp[0] * scale) & 0xff;
        int g = (int)(tp[sd] * scale) & 0xff;
        int b = (int)(tp[sd+sd] * scale) & 0xff;
        tp += sw;
        ip[i] = qRgb(r,g,b);
      }
    }
    else if (depth == 4)
    {
      for (int i=0; i<width; i++)
      {
        int a = (int)(tp[sd+sd+sd] * scale) & 0xff;
        int r = (int)(tp[0] * scale) & 0xff;
        int g = (int)(tp[sd] * scale) & 0xff;
        int b = (int)(tp[sd+sd] * scale) & 0xff;
        tp += sw;
        ip[i] = qRgba(r,g,b,a);
      }
    }
  }
  // return
  luaQ_pushqt(L, image);
  return 1;
}

static int qttorch_(qimage_totensor)(lua_State *L)
{
  THTensor *Tdst = 0;
  QImage image = luaQ_checkqvariant<QImage>(L, 1);
  int width = image.width();
  int height = image.height();
  int depth = 1;
  int ddim, wdim = 1, hdim = 0;
  int tpos = 0;
  double scale = 255.0 / luaL_optnumber(L, 3, 1);

  // validate arguments
  if (lua_type(L, 2) == LUA_TUSERDATA)
  {
    tpos = 2;
    Tdst = (THTensor*)luaT_checkudata(L,2,torch_Tensor);
    if (Tdst->nDimension == 3)
    {
      ddim = 0;
      hdim = 1;
      wdim = 2;
      depth = Tdst->size[ddim];
    }
    else if (Tdst->nDimension != 2)
      luaL_error(L, "tensor must have 2 or 3 dimensions");
    if (depth != 1 && depth != 3 && depth != 4)
      luaL_error(L, "tensor third dimension must be 1, 3, or 4.");
    if (width != Tdst->size[wdim] || height != Tdst->size[hdim])
      luaL_error(L, "tensor dimensions must match the image size.");
  }
  else
  {
    depth = luaL_optinteger(L, 2, 3);
    if (depth != 1 && depth != 3 && depth != 4)
      luaL_error(L, "depth must be 1, 3, or 4.");
    if (depth == 1)
      Tdst = THTensor_(newWithSize2d)(height, width);
    else
    {
      ddim = 0;
      hdim = 1;
      wdim = 2;
      Tdst = THTensor_(newWithSize3d)(depth, height, width);
    }
  }

  // convert image
  if (image.format() != QImage::Format_ARGB32)
    image = image.convertToFormat(QImage::Format_ARGB32);
  if (image.format() != QImage::Format_ARGB32)
    luaL_error(L, "Cannot convert image to format ARGB32");

  // fill tensor
  long sw = Tdst->stride[wdim];
  long sh = Tdst->stride[hdim];
  long sd = (depth > 1) ? Tdst->stride[ddim] : 0;
  real *tdata = THTensor_(data)(Tdst);
  for(int j=0; j<height; j++) 
  {
    QRgb *ip = (QRgb*)image.scanLine(j);
    real *tp = tdata + sh * j;
    if (depth == 1)
    {
      for (int i=0; i<width; i++)
      {
        QRgb v = ip[i];
        tp[0] = (real)qGray(v) / scale;
        tp += sw;
      }
    }
    else if (depth == 3)
    {
      for (int i=0; i<width; i++)
      {
        QRgb v = ip[i];
        tp[0] = (real)qRed(v) / scale;
        tp[sd] = (real)qGreen(v) / scale;
        tp[sd+sd] = (real)qBlue(v) / scale;
        tp += sw;
      }
    }
    else if (depth == 4)
    {
      for (int i=0; i<width; i++)
      {
        QRgb v = ip[i];
        tp[0] = (real)qRed(v) / scale;
        tp[sd] = (real)qGreen(v) / scale;
        tp[sd+sd] = (real)qBlue(v) / scale;
        tp[sd+sd+sd] = (real)qAlpha(v) / scale;
        tp += sw;
      }
    }
  }
  // return
  if (tpos > 0)
    lua_pushvalue(L, tpos);
  else
    luaT_pushudata(L, (void*)Tdst, torch_Tensor);
  return 1;
}


static struct luaL_Reg qttorch_(qimage_lib)[] = {
  {"QImageFromTensor", qttorch_(qimage_fromtensor)},
  {"QImageToTensor", qttorch_(qimage_totensor)},
  {0,0}
};

static void qttorch_(Tensor_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, qttorch_(qimage_lib), "qttorch");
  lua_pop(L,1);
}

#endif

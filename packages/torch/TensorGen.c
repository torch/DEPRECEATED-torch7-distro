/* Storage type */
#define STORAGE_T_(TYPE) TH##TYPE##Storage
#define STORAGE_T(TYPE) STORAGE_T_(TYPE)
#define STORAGE STORAGE_T(CAP_TYPE)

/* Tensor type */
#define TENSOR_T_(TYPE) TH##TYPE##Tensor
#define TENSOR_T(TYPE) TENSOR_T_(TYPE)
#define TENSOR TENSOR_T(CAP_TYPE)

/* Name in Lua */
#define LUA_TENSOR_NAME__(TYPE) "torch." #TYPE "Tensor"
#define LUA_TENSOR_NAME_(TYPE) LUA_TENSOR_NAME__(TYPE)
#ifdef DEFAULT_TENSOR
#define LUA_TENSOR_NAME "torch.Tensor"
#else
#define LUA_TENSOR_NAME LUA_TENSOR_NAME_(CAP_TYPE)
#endif

/* Function name for a Storage */
#define STORAGE_FUNC_TN_(TYPE,NAME) TH##TYPE##Storage_##NAME
#define STORAGE_FUNC_TN(TYPE, NAME) STORAGE_FUNC_TN_(TYPE,NAME) 
#define STORAGE_FUNC(NAME) STORAGE_FUNC_TN(CAP_TYPE, NAME)

/* Function name for a Tensor */
#define TENSOR_FUNC_TN_(TYPE,NAME) TH##TYPE##Tensor_##NAME
#define TENSOR_FUNC_TN(TYPE, NAME) TENSOR_FUNC_TN_(TYPE,NAME) 
#define TENSOR_FUNC(NAME) TENSOR_FUNC_TN(CAP_TYPE, NAME)

/* Wrapper function name for a Tensor */
#define W_TENSOR_FUNC_TN_(TYPE, NAME) torch_##TYPE##Tensor_##NAME
#define W_TENSOR_FUNC_TN(TYPE, NAME) W_TENSOR_FUNC_TN_(TYPE, NAME)
#define W_TENSOR_FUNC(NAME) W_TENSOR_FUNC_TN(CAP_TYPE, NAME)

/* Storage id in Lua */
#define LUA_STORAGE_T_(TYPE) torch_##TYPE##Storage_id
#define LUA_STORAGE_T(TYPE) LUA_STORAGE_T_(TYPE)
#define LUA_STORAGE LUA_STORAGE_T(CAP_TYPE)

/* Tensor id in Lua */
#define LUA_TENSOR W_TENSOR_FUNC(id)

/* For the default Tensor type, we simplify the naming */
#ifdef DEFAULT_TENSOR
#undef TENSOR
#undef TENSOR_FUNC
#undef W_TENSOR_FUNC
#define TENSOR THTensor
#define TENSOR_FUNC(NAME) TENSOR_FUNC_TN(, NAME)
#define W_TENSOR_FUNC(NAME) W_TENSOR_FUNC_TN(, NAME)
#endif

static void W_TENSOR_FUNC(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         STORAGE **storage_, long *storageOffset_, int *nDimension_, long **size_, long **stride_);

static void W_TENSOR_FUNC(c_readSizeStride)(lua_State *L, int index, int allowStride, int *nDimension_, long **size_, long **stride_);

static int W_TENSOR_FUNC(size)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->size[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->size, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, torch_LongStorage_id);
  }
  return 1;
}

static int W_TENSOR_FUNC(stride)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->stride[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->stride, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, torch_LongStorage_id);
  }
  return 1;
}

static int W_TENSOR_FUNC(nDimension)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  lua_pushnumber(L, tensor->nDimension);
  return 1;
}

static int W_TENSOR_FUNC(storage)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  STORAGE_FUNC(retain)(tensor->storage);
  luaT_pushudata(L, tensor->storage, LUA_STORAGE);
  return 1;
}

static int W_TENSOR_FUNC(storageOffset)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  lua_pushnumber(L, tensor->storageOffset+1);
  return 1;
}

static int W_TENSOR_FUNC(ownStorage)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  lua_pushboolean(L, tensor->ownStorage);
  return 1;
}

static int W_TENSOR_FUNC(new)(lua_State *L)
{
  TENSOR *tensor;
  STORAGE *storage;
  long storageOffset;
  int nDimension;
  long *size, *stride;

  W_TENSOR_FUNC(c_readTensorStorageSizeStride)(L, 1, 1, 1, 1, 1,
                                               &storage, &storageOffset, &nDimension, &size, &stride);

  tensor = TENSOR_FUNC(newWithStorage)(storage, storageOffset, nDimension, size, stride);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

static int W_TENSOR_FUNC(set)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  STORAGE *storage;
  long storageOffset;
  int nDimension;
  long *size, *stride;

  W_TENSOR_FUNC(c_readTensorStorageSizeStride)(L, 2, 1, 1, 1, 1,
                                               &storage, &storageOffset, &nDimension, &size, &stride);

  TENSOR_FUNC(setStorage)(tensor, storage, storageOffset, nDimension, size, stride);
  lua_settop(L, 1);
  return 1;
}


/* Resize */
static int W_TENSOR_FUNC(resizeAs)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  TENSOR *src = luaT_checkudata(L, 2, LUA_TENSOR);
  TENSOR_FUNC(resizeAs)(tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int W_TENSOR_FUNC(resize)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  long *size, *stride;
  int nDimension;

  W_TENSOR_FUNC(c_readSizeStride)(L, 2, 0, &nDimension, &size, &stride);

  TENSOR_FUNC(resize)(tensor, nDimension, size);
  lua_settop(L, 1);
  return 1;
}

static int W_TENSOR_FUNC(narrow)(lua_State *L)
{
  TENSOR *src = luaT_checkudata(L, 1, LUA_TENSOR);
  int dimension = luaL_checkint(L, 2)-1;
  long firstIndex = luaL_checklong(L, 3)-1;
  long size = luaL_checklong(L, 4);
  TENSOR *tensor;

/*  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 4, "out of range");
*/
  tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(narrow)(tensor, src, dimension, firstIndex, size);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

static int W_TENSOR_FUNC(sub)(lua_State *L)
{
  TENSOR *src = luaT_checkudata(L, 1, LUA_TENSOR);
  TENSOR *tensor;
  long d0s = -1, d0e = -1, d1s = -1, d1e = -1, d2s = -1, d2e = -1, d3s = -1, d3e = -1;

  d0s = luaL_checklong(L, 2)-1;
  d0e = luaL_checklong(L, 3)-1;
  if(d0s < 0)
    d0s += src->size[0]+1;
  if(d0e < 0)
    d0e += src->size[0]+1;
  luaL_argcheck(L, src->nDimension > 0, 2, "invalid dimension");
  luaL_argcheck(L, d0s >= 0 && d0s < src->size[0], 2, "out of range");
  luaL_argcheck(L, d0e >= 0 && d0e < src->size[0], 3, "out of range");
  luaL_argcheck(L, d0e >= d0s, 3, "end smaller than beginning");

  if(!lua_isnone(L, 4))
  {
    d1s = luaL_checklong(L, 4)-1;
    d1e = luaL_checklong(L, 5)-1;
    if(d1s < 0)
      d1s += src->size[1]+1;
    if(d1e < 0)
      d1e += src->size[1]+1;
    luaL_argcheck(L, src->nDimension > 1, 4, "invalid dimension");
    luaL_argcheck(L, d1s >= 0 && d1s < src->size[1], 4, "out of range");
    luaL_argcheck(L, d1e >= 0 && d1e < src->size[1], 5, "out of range");    
    luaL_argcheck(L, d1e >= d1s, 5, "end smaller than beginning");

    if(!lua_isnone(L, 6))
    {
      d2s = luaL_checklong(L, 6)-1;
      d2e = luaL_checklong(L, 7)-1;
      if(d2s < 0)
        d2s += src->size[2]+1;
      if(d2e < 0)
        d2e += src->size[2]+1;
      luaL_argcheck(L, src->nDimension > 2, 6, "invalid dimension");
      luaL_argcheck(L, d2s >= 0 && d2s < src->size[2], 6, "out of range");
      luaL_argcheck(L, d2e >= 0 && d2e < src->size[2], 7, "out of range");    
      luaL_argcheck(L, d2e >= d2s, 7, "end smaller than beginning");

      if(!lua_isnone(L, 8))
      {
        d3s = luaL_checklong(L, 8)-1;
        d3e = luaL_checklong(L, 9)-1;
        if(d3s < 0)
          d3s += src->size[3]+1;
        if(d3e < 0)
          d3e += src->size[3]+1;
        luaL_argcheck(L, src->nDimension > 3, 8, "invalid dimension");
        luaL_argcheck(L, d3s >= 0 && d3s < src->size[3], 8, "out of range");
        luaL_argcheck(L, d3e >= 0 && d3e < src->size[3], 9, "out of range");    
        luaL_argcheck(L, d3e >= d3s, 9, "end smaller than beginning");
      }
    }
  }

  tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(narrow)(tensor, src, 0, d0s, d0e-d0s+1);
  if(d1s >= 0)
    TENSOR_FUNC(narrow)(tensor, NULL, 1, d1s, d1e-d1s+1);
  if(d2s >= 0)
    TENSOR_FUNC(narrow)(tensor, NULL, 2, d2s, d2e-d2s+1);
  if(d3s >= 0)
    TENSOR_FUNC(narrow)(tensor, NULL, 3, d3s, d3e-d3s+1);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

static int W_TENSOR_FUNC(select)(lua_State *L)
{
  TENSOR *src = luaT_checkudata(L, 1, LUA_TENSOR);
  int dimension = luaL_checkint(L, 2)-1;
  long sliceIndex = luaL_checklong(L, 3)-1;
  TENSOR *tensor;

/*   THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");
*/

  tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(select)(tensor, src, dimension, sliceIndex);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}


static int W_TENSOR_FUNC(transpose)(lua_State *L)
{
  TENSOR *src = luaT_checkudata(L, 1, LUA_TENSOR);
  int dimension1 = luaL_checkint(L, 2)-1;
  int dimension2 = luaL_checkint(L, 3)-1;
  TENSOR *tensor;

/*
  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 2, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 3, "out of range");
*/

  tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(transpose)(tensor, src, dimension1, dimension2);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

static int W_TENSOR_FUNC(t)(lua_State *L)
{
  TENSOR *src = luaT_checkudata(L, 1, LUA_TENSOR);
  TENSOR *tensor;

  luaL_argcheck(L, src->nDimension == 2, 1, "Tensor must have 2 dimensions");

  tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(transpose)(tensor, src, 0, 1);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

int W_TENSOR_FUNC(unfold)(lua_State *L)
{
  TENSOR *src = luaT_checkudata(L, 1, LUA_TENSOR);
  int dimension = luaL_checkint(L, 2)-1;
  long size = luaL_checklong(L, 3);
  long step = luaL_checklong(L, 4);
  TENSOR *tensor;

/*
  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
*/

  tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(unfold)(tensor, src, dimension, size, step);
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

/* fill */
static int W_TENSOR_FUNC(fill)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  TYPE value = (TYPE)luaL_checknumber(L, 2);
  TENSOR_FUNC(fill)(tensor, value);
  lua_settop(L, 1);
  return 1;
}

/* is contiguous? [a bit like in TnXIterator] */
static int W_TENSOR_FUNC(isContiguous)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  lua_pushboolean(L, TENSOR_FUNC(isContiguous)(tensor));
  return 1;
}

static int W_TENSOR_FUNC(nElement)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  lua_pushnumber(L, TENSOR_FUNC(nElement)(tensor));
  return 1;
}

static int W_TENSOR_FUNC(copy)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  void *src;
  if( (src = luaT_toudata(L, 2, LUA_TENSOR)) )
    TENSOR_FUNC(copy)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_ByteTensor_id)) )
    TENSOR_FUNC(copyByte)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_CharTensor_id)) )
    TENSOR_FUNC(copyChar)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_ShortTensor_id)) )
    TENSOR_FUNC(copyShort)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_IntTensor_id)) )
    TENSOR_FUNC(copyInt)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_LongTensor_id)) )
    TENSOR_FUNC(copyLong)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_FloatTensor_id)) )
    TENSOR_FUNC(copyFloat)(tensor, src);
  else if( (src = luaT_toudata(L, 2, torch_Tensor_id)) )
    TENSOR_FUNC(copyDouble)(tensor, src);
  else
    luaL_typerror(L, 2, "torch.*Tensor");
  lua_settop(L, 1);
  return 1;
}

static int W_TENSOR_FUNC(__newindex__)(lua_State *L)
{
  if(lua_isnumber(L, 2))
  {
    TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
    long index = luaL_checklong(L,2)-1;
    TYPE value = (TYPE)luaL_checknumber(L,3);
    luaL_argcheck(L, tensor->nDimension == 1, 1, "must be a one dimensional tensor");
    luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");
    (tensor->storage->data+tensor->storageOffset)[index*tensor->stride[0]] = value;
    lua_pushboolean(L, 1);
  }
  else
    lua_pushboolean(L, 0);

  return 1;
}

static int W_TENSOR_FUNC(__index__)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);

  if(lua_isnumber(L, 2))
  {
    long index = luaL_checklong(L,2)-1;
    
    luaL_argcheck(L, tensor->nDimension > 0, 1, "empty tensor");
    luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");

    if(tensor->nDimension == 1)
    {
      lua_pushnumber(L, (tensor->storage->data+tensor->storageOffset)[index*tensor->stride[0]]);
    }
    else
    {
      TENSOR *tensor_ = TENSOR_FUNC(new)();
      TENSOR_FUNC(select)(tensor_, tensor, 0, index);
      luaT_pushudata(L, tensor_, LUA_TENSOR);
    }
    lua_pushboolean(L, 1);
    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

static int W_TENSOR_FUNC(free)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  TENSOR_FUNC(free)(tensor);
  return 0;
}

/* helpful functions */
static void W_TENSOR_FUNC(c_readSizeStride)(lua_State *L, int index, int allowStride, int *nDimension_, long **size_, long **stride_)
{
  static long isize[4];
  static long istride[4];
  THLongStorage *size = NULL;
  THLongStorage *stride = NULL;
  
  if( (size = luaT_toudata(L, index, torch_LongStorage_id)) )
  {
    if(!lua_isnoneornil(L, index+1))
    {
      if( (stride = luaT_toudata(L, index+1, torch_LongStorage_id)) )
        luaL_argcheck(L, stride->size == size->size, index+1, "provided stride and size are inconsistent");
      else
        luaL_argcheck(L, 0, index+1, "torch.LongStorage expected");
    }
    *nDimension_ = size->size;
    *size_ = size->data;
    *stride_ = (stride ? stride->data : NULL);
  }
  else
  {
    int i;
    for(i = 0; i < 4; i++)
    {
      isize[i] = -1;
      istride[i] = -1;
    }
    if(allowStride)
    {
      for(i = 0; i < 4; i++)
      {
        if(lua_isnone(L, index+2*i))
          break;
        isize[i] = luaL_checklong(L, index+2*i);
        
        if(lua_isnone(L, index+2*i+1))
          break;
        istride[i] = luaL_checklong(L, index+2*i+1);
      }
    }
    else
    {
      for(i = 0; i < 4; i++)
      {
        if(lua_isnone(L, index+i))
          break;
        isize[i] = luaL_checklong(L, index+i);
      }
    }
    *nDimension_ = 4;
    *size_ = isize;
    *stride_ = istride;
  }
}

static void W_TENSOR_FUNC(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         STORAGE **storage_, long *storageOffset_, int *nDimension_, long **size_, long **stride_)
{
  static char errMsg[64];
  TENSOR *src = NULL;
  STORAGE *storage = NULL;

  int arg1Type = lua_type(L, index);

  if( allowNone && (arg1Type == LUA_TNONE) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    *nDimension_ = 0;
    *size_ = NULL;
    *stride_ = NULL;
    return;
  }
  else if( allowTensor && (arg1Type == LUA_TUSERDATA) && (src = luaT_toudata(L, index, LUA_TENSOR)) )
  {
    *storage_ = src->storage;
    *storageOffset_ = src->storageOffset;
    *nDimension_ = src->nDimension;
    *size_ = src->size;
    *stride_ = src->stride;
    return;
  }
  else if( allowStorage && (arg1Type == LUA_TUSERDATA) && (storage = luaT_toudata(L, index, LUA_STORAGE)) )
  {
    *storage_ = storage;
    if(lua_isnone(L, index+1))
    {
      static long __stride__ = 1;
      *storageOffset_ = 0;
      *nDimension_ = 1;
      *size_ = &storage->size;
      *stride_ = &__stride__;
    }
    else
    {
      *storageOffset_ = luaL_checklong(L, index+1)-1;
      W_TENSOR_FUNC(c_readSizeStride)(L, index+2, allowStride, nDimension_, size_, stride_);
    }
    return;
  }
  else if( (arg1Type == LUA_TNUMBER) || (luaT_toudata(L, index, torch_LongStorage_id)) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    W_TENSOR_FUNC(c_readSizeStride)(L, index, 0, nDimension_, size_, stride_);

    return;
  }
  sprintf(errMsg, "expecting number%s%s", (allowTensor ? " or Tensor" : ""), (allowStorage ? " or Storage" : ""));
  luaL_argcheck(L, 0, index, errMsg);
}

static int W_TENSOR_FUNC(apply)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  luaL_checktype(L, 2, LUA_TFUNCTION);
  lua_settop(L, 2);

  TH_TENSOR_APPLY(TYPE, tensor,
                  lua_pushvalue(L, 2);
                  lua_pushnumber(L, *tensor_p);
                  lua_call(L, 1, 1);
                  if(lua_isnumber(L, 3))
                  {
                    *tensor_p = (TYPE)lua_tonumber(L, 3);
                    lua_pop(L, 1);
                  }
                  else if(lua_isnil(L, 3))
                    lua_pop(L, 1);
                  else
                    luaL_error(L, "given function should return a number or nil"););

  lua_settop(L, 1);
  return 1;
}

static int W_TENSOR_FUNC(map)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  TENSOR *src = luaT_checkudata(L, 2, LUA_TENSOR);
  luaL_checktype(L, 3, LUA_TFUNCTION);
  lua_settop(L, 3);

  TH_TENSOR_APPLY2(TYPE, tensor, TYPE, src,
                  lua_pushvalue(L, 3);
                  lua_pushnumber(L, *tensor_p);
                  lua_pushnumber(L, *src_p);
                  lua_call(L, 2, 1);
                  if(lua_isnumber(L, 4))
                  {
                    *tensor_p = (TYPE)lua_tonumber(L, 4);
                    lua_pop(L, 1);
                  }
                  else if(lua_isnil(L, 4))
                    lua_pop(L, 1);
                  else
                    luaL_error(L, "given function should return a number or nil"););

  lua_settop(L, 1);
  return 1;
}

static int W_TENSOR_FUNC(map2)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  TENSOR *src1 = luaT_checkudata(L, 2, LUA_TENSOR);
  TENSOR *src2 = luaT_checkudata(L, 3, LUA_TENSOR);
  luaL_checktype(L, 4, LUA_TFUNCTION);
  lua_settop(L, 4);

  TH_TENSOR_APPLY3(TYPE, tensor, TYPE, src1, TYPE, src2,
                  lua_pushvalue(L, 4);
                  lua_pushnumber(L, *tensor_p);
                  lua_pushnumber(L, *src1_p);
                  lua_pushnumber(L, *src2_p);
                  lua_call(L, 3, 1);
                  if(lua_isnumber(L, 5))
                  {
                    *tensor_p = (TYPE)lua_tonumber(L, 5);
                    lua_pop(L, 1);
                  }
                  else if(lua_isnil(L, 5))
                    lua_pop(L, 1);
                  else
                    luaL_error(L, "given function should return a number or nothing"););

  lua_settop(L, 1);
  return 1;
}

static int W_TENSOR_FUNC(factory)(lua_State *L)
{
  TENSOR *tensor = TENSOR_FUNC(new)();
  luaT_pushudata(L, tensor, LUA_TENSOR);
  return 1;
}

static int W_TENSOR_FUNC(write)(lua_State *L)
{  
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  long storageOffset = tensor->storageOffset+1; /* to respect Lua convention */

  lua_pushvalue(L, 2);
  torch_File_writeInt(L, &tensor->nDimension, 1);
  torch_File_writeLong(L, tensor->size, tensor->nDimension);
  torch_File_writeLong(L, tensor->stride, tensor->nDimension);
  torch_File_writeLong(L, &storageOffset, 1);
  torch_File_writeInt(L, &tensor->ownStorage, 1);
  STORAGE_FUNC(retain)(tensor->storage);
  luaT_pushudata(L, tensor->storage, LUA_STORAGE);
  torch_File_writeObject(L);
  return 0;
}

static int W_TENSOR_FUNC(read)(lua_State *L)
{
  TENSOR *tensor = luaT_checkudata(L, 1, LUA_TENSOR);
  int version = luaL_checkint(L, 3);
  long storageOffset;
  
  lua_pushvalue(L, 2);
  torch_File_readInt(L, &tensor->nDimension, 1);
  tensor->size = THAlloc(sizeof(long)*tensor->nDimension);
  tensor->stride = THAlloc(sizeof(long)*tensor->nDimension);
  if(version > 0)
  {
    torch_File_readLong(L, tensor->size, tensor->nDimension);
    torch_File_readLong(L, tensor->stride, tensor->nDimension);
    torch_File_readLong(L, &storageOffset, 1);
  }
  else
  {
    int *buffer_ = THAlloc(sizeof(int)*tensor->nDimension);
    int storageOffset_;
    int i;

    torch_File_readInt(L, buffer_, tensor->nDimension);
    for(i = 0; i < tensor->nDimension; i++)
      tensor->size[i] = buffer_[i];

    torch_File_readInt(L, buffer_, tensor->nDimension);
    for(i = 0; i < tensor->nDimension; i++)
      tensor->stride[i] = buffer_[i];

    torch_File_readInt(L, &storageOffset_, 1);
    storageOffset = storageOffset_;
    THFree(buffer_);
  }
  tensor->storageOffset = storageOffset-1; /* to respect Lua convention */
  torch_File_readInt(L, &tensor->ownStorage, 1);
  torch_File_readObject(L);  
  tensor->storage = luaT_toudata(L, -1, LUA_STORAGE);
  STORAGE_FUNC(retain)(tensor->storage);
  lua_pop(L, 1);
  return 0;
}

static const struct luaL_Reg W_TENSOR_FUNC(_) [] = {
  {"size", W_TENSOR_FUNC(size)},
  {"__len__", W_TENSOR_FUNC(size)},
  {"stride", W_TENSOR_FUNC(stride)},
  {"dim", W_TENSOR_FUNC(nDimension)},
  {"nDimension", W_TENSOR_FUNC(nDimension)},
  {"storage", W_TENSOR_FUNC(storage)},
  {"storageOffset", W_TENSOR_FUNC(storageOffset)},
  {"ownStorage", W_TENSOR_FUNC(ownStorage)},
  {"set", W_TENSOR_FUNC(set)},
  {"resizeAs", W_TENSOR_FUNC(resizeAs)},
  {"resize", W_TENSOR_FUNC(resize)},
  {"narrow", W_TENSOR_FUNC(narrow)},
  {"sub", W_TENSOR_FUNC(sub)},
  {"select", W_TENSOR_FUNC(select)},
  {"transpose", W_TENSOR_FUNC(transpose)},
  {"t", W_TENSOR_FUNC(t)},
  {"unfold", W_TENSOR_FUNC(unfold)},
  {"fill", W_TENSOR_FUNC(fill)},
  {"isContiguous", W_TENSOR_FUNC(isContiguous)},
  {"nElement", W_TENSOR_FUNC(nElement)},
  {"copy", W_TENSOR_FUNC(copy)},
  {"apply", W_TENSOR_FUNC(apply)},
  {"map", W_TENSOR_FUNC(map)},
  {"map2", W_TENSOR_FUNC(map2)},
  {"read", W_TENSOR_FUNC(read)},
  {"write", W_TENSOR_FUNC(write)},
  {"__index__", W_TENSOR_FUNC(__index__)},
  {"__newindex__", W_TENSOR_FUNC(__newindex__)},
  {NULL, NULL}
};

void W_TENSOR_FUNC(init)(lua_State *L)
{
  torch_ByteStorage_id = luaT_checktypename2id(L, "torch.ByteStorage");
  torch_CharStorage_id = luaT_checktypename2id(L, "torch.CharStorage");
  torch_ShortStorage_id = luaT_checktypename2id(L, "torch.ShortStorage");
  torch_IntStorage_id = luaT_checktypename2id(L, "torch.IntStorage");
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");
  torch_FloatStorage_id = luaT_checktypename2id(L, "torch.FloatStorage");
  torch_DoubleStorage_id = luaT_checktypename2id(L, "torch.DoubleStorage");

  LUA_TENSOR = luaT_newmetatable(L, LUA_TENSOR_NAME, NULL,
                                 W_TENSOR_FUNC(new), W_TENSOR_FUNC(free), W_TENSOR_FUNC(factory));
  luaL_register(L, NULL, W_TENSOR_FUNC(_));
  lua_pop(L, 1);
}

#undef LUA_TENSOR_NAME

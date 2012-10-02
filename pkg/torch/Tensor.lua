ffi.cdef([[

typedef struct THTensor
{
    long *__size;
    long *__stride;
    int __nDimension;
    
    THStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THTensor;


THTensor *THTensor_new(void);
THTensor *THTensor_newWithStorage(THStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THTensor *THTensor_newClone(THTensor *self);
THTensor *THTensor_newContiguous(THTensor *tensor);
THTensor *THTensor_newSelect(THTensor *tensor, int dimension_, long sliceIndex_);
THTensor *THTensor_newNarrow(THTensor *tensor, int dimension_, long firstIndex_, long size_);
THTensor *THTensor_newTranspose(THTensor *tensor, int dimension1_, int dimension2_);
THTensor *THTensor_newUnfold(THTensor *tensor, int dimension_, long size_, long step_);

void THTensor_resize(THTensor *tensor, THLongStorage *size, THLongStorage *stride);
void THTensor_resizeAs(THTensor *tensor, THTensor *src);

void THTensor_squeeze(THTensor *self, THTensor *src);
void THTensor_squeeze1d(THTensor *self, THTensor *src, int dimension_);
    
int THTensor_isContiguous(const THTensor *self);
long THTensor_nElement(const THTensor *self);

void THTensor_free(THTensor *self);

real THTensor_get1d(const THTensor *tensor, long x0);
real THTensor_get2d(const THTensor *tensor, long x0, long x1);
real THTensor_get3d(const THTensor *tensor, long x0, long x1, long x2);
real THTensor_get4d(const THTensor *tensor, long x0, long x1, long x2, long x3);

]])

-- checkout http://www.torch.ch/manual/torch/tensor
local function readtensorsizestride(arg)
   local storage
   local offset
   local size
   local stride
   local narg = #arg

   if narg == 0 then
      return nil, 0, nil, nil
   elseif narg == 1 and type(arg[1]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1]}, nil
   elseif narg == 1 and type(arg[1]) == 'table' then
      error('not implemented yet')
      -- todo
   elseif narg == 1 and type(arg[1]) == 'torch.LongStorage' then
      return nil, 0, arg[1], nil
   elseif narg == 1 and type(arg[1]) == 'torch.Storage' then
      return arg[1], 0, nil, nil
   elseif narg == 1 and type(arg[1]) == 'torch.Tensor' then
      return arg[1]:storage(), arg[1]:storageOffset(), arg[1]:size(), arg[1]:stride()
   elseif narg == 2 and type(arg[1]) == 'number' and type(arg[2]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1], arg[2]}, nil
   elseif narg == 2 and type(arg[1]) == 'torch.LongStorage' and type(arg[2]) == 'torch.LongStorage' then
      return nil, 0, arg[1], arg[2]
   elseif narg == 3 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1], arg[2], arg[3]}
   elseif narg == 3 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number' and type(arg[3]) == 'torch.LongStorage' then
      return arg[1], arg[2], arg[3], nil
   elseif narg == 3 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number' and type(arg[3]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3]}, nil
   elseif narg == 4 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' and type(arg[4]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1], arg[2], arg[3], arg[4]}
   elseif narg == 4 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number' and type(arg[3]) == 'torch.LongStorage' and type(arg[4]) == 'torch.LongStorage' then
      return arg[1], arg[2], arg[3], arg[4]
   elseif narg == 4 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3]}, torch.LongStorage{arg[4]}
   elseif narg == 5 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5]}, torch.LongStorage{arg[4]}
   elseif narg == 6 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5]}, torch.LongStorage{arg[4], arg[6]}
   elseif narg == 7 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7]}, torch.LongStorage{arg[4], arg[6]}
   elseif narg == 8 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' and type(arg[8]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7]}, torch.LongStorage{arg[4], arg[6], arg[8]}
   elseif narg == 9 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' and type(arg[8]) == 'number' and type(arg[9]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7], arg[9]}, torch.LongStorage{arg[4], arg[6], arg[8]}
   elseif narg == 10 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' and type(arg[8]) == 'number' and type(arg[9]) == 'number' and type(arg[10]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7], arg[9]}, torch.LongStorage{arg[4], arg[6], arg[8], arg[10]}
   else
      error('invalid arguments')
   end
end

local function readsizestride(arg)
   local size
   local stride
   local narg = #arg

   if narg == 1 and type(arg[1]) == 'number' then
      return torch.LongStorage{arg[1]}, nil
   elseif narg == 1 and type(arg[1]) == 'table' then
      return torch.LongStorage(arg[1]), nil
   elseif narg == 1 and type(arg[1]) == 'torch.LongStorage' then
      return arg[1], nil
   elseif narg == 2 and type(arg[1]) == 'number' and type(arg[2]) == 'number' then
      return torch.LongStorage{arg[1], arg[2]}, nil
   elseif narg == 2 and type(arg[1]) == 'table' and type(arg[2]) == 'table' then
      return torch.LongStorage(arg[1]), torch.LongStorage(arg[2])
   elseif narg == 2 and type(arg[1]) == 'torch.LongStorage' and type(arg[2]) == 'torch.LongStorage' then
      return arg[1], arg[2]
   elseif narg == 3 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' then
      return torch.LongStorage{arg[1], arg[2], arg[3]}, nil
   elseif narg == 4 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' and type(arg[4]) == 'number' then
      return torch.LongStorage{arg[1], arg[2], arg[3], arg[4]}, nil
   else
      error('invalid arguments')
   end
end

local mt = {
   __typename = "torch.Tensor",

   nDimension = function(self)
                   return self.__nDimension
                end,

   storage = function(self)
                return self.__storage[0] -- DEBUG: what about retaining here?
             end,

   storageOffset = function(self)
                      return tonumber(self.__storageOffset)
                   end,

   size = function(self, dim)
             if dim then
                if dim > 0 and dim <= self.__nDimension then
                   return tonumber(self.__size[dim-1])
                else
                   error('out of bounds')
                end
             else
                return torch.LongStorage(self.__nDimension):rawCopy(self.__size)
             end
          end,

   resize = function(self, ...)
               local arg = {...}
               local size, stride = readsizestride(arg)
               TH.THTensor_resize(self, size, stride)
               return self
            end,

   new = function(...)
            local self
            local arg = {...}
            local storage, offset, size, stride = readtensorsizestride(arg)
            self = TH.THTensor_newWithStorage(storage, offset, size, stride)[0]
            ffi.gc(self, TH.THTensor_free)
            return self
         end
}

ffi.metatype("THTensor", {__index=function(self, k)
                                     if type(k) == 'number' then
                                        if self.__nDimension == 1 then
                                           return tonumber(TH.THTensor_get1d(self, k-1))
                                        elseif self.__nDimension > 1 then
                                           return TH.THTensor_newSelect(self, 0, k-1)
                                        else
                                           error('empty tensor')
                                        end
                                     else
                                        return mt[k]
                                     end
                                  end})


torch.Tensor = {}
setmetatable(torch.Tensor, {__index=mt,
                            __metatable=mt,
                            __newindex=mt,
                            __call=function(self, ...)
                                      return mt.new(...)
                                   end})

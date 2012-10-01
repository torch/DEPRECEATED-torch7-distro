ffi.cdef([[

typedef struct THTensor
{
    long *size;
    long *stride;
    int nDimension;
    
    THStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

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

local mt = {

   size = function(self)
             return TH.THTensor_size(self.core)
          end,

   resize = function(self, size)
               return TH.THTensor_resize(self.core, size)
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
                                        if k > 0 and k <= self.size then
                                           return self.data[k-1]
                                        else
                                           error('bound')
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

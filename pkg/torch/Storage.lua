ffi.cdef([[

typedef struct
{
    real *__data;
    long __size;
    int __refcount;
    char __flag;   
} THStorage;

THStorage* THStorage_new(void);
THStorage* THStorage_newWithSize(long size);
THStorage* THStorage_newWithMapping(const char *fileName, int isShared);
long THStorage_size(const THStorage*);
real* THStorage_data(const THStorage*);
void THStorage_fill(THStorage* storage, real value);
real THStorage_get(const THStorage* storage, long idx);
void THStorage_set(const THStorage* storage, long idx, real value);
void THStorage_resize(THStorage* storage, long size);
void THStorage_free(THStorage *storage);

]])


local mt = {
   __typename = "torch.Storage",

   fill = function(self, value)
             TH.THStorage_fill(self, value)
             return self
          end,

   size = function(self)
             return tonumber(self.__size)
          end,

   resize = function(self, size)
               TH.THStorage_resize(self, size)
               return self
            end,

   new = function(...)
            local arg = {...}
            local narg = #arg
            local self
            if narg == 0 then
               self = TH.THStorage_new()[0]
            elseif narg == 1 and type(arg[1]) == 'number' then
               self = TH.THStorage_newWithSize(arg[1])[0]
            elseif narg == 1 and type(arg[1]) == 'table' then
               local tbl = arg[1]
               local size = #tbl
               self = TH.THStorage_newWithSize(size)[0]
               for i=1,size do
                  self.__data[i-1] = tbl[i]
               end
            elseif narg == 1 and type(arg[1]) == 'string' then
               self = TH.THStorage_newWithMapping(arg[1], 0)[0]
            elseif narg == 2 and type(arg[1]) == 'string' and type(arg[2]) == 'boolean' then
               self = TH.THStorage_newWithMapping(arg[1], arg[2])[0]
            else
               error('invalid arguments')
            end
            ffi.gc(self, TH.THStorage_free)
            return self
         end
}

ffi.metatype("THStorage", {__index=function(self, k)
                                      if type(k) == 'number' then
                                         if k > 0 and k <= self.__size then
                                            return tonumber(self.__data[k-1])
                                         else
                                            error('index out of bounds')
                                         end
                                      else
                                         return mt[k]
                                      end
                                   end,

                           __newindex=function(self, k, v)
                                         if k > 0 and k <= self.__size then
                                            self.__data[k-1] = v
                                         else
                                            error('index out of bounds')
                                         end
                                      end})

torch.Storage = {}
setmetatable(torch.Storage, {__index=mt,
                             __metatable=mt,
                             __newindex=mt,
                             __call=function(self, ...)
                                       return mt.new(...)
                                    end})


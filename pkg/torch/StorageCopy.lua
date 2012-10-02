ffi.cdef([[
void THStorage_rawCopy(THStorage *storage, real *src);
void THStorage_copy(THStorage *storage, THStorage *src);
void THStorage_copyByte(THStorage *storage, THByteStorage *src);
void THStorage_copyChar(THStorage *storage, THCharStorage *src);
void THStorage_copyShort(THStorage *storage, THShortStorage *src);
void THStorage_copyInt(THStorage *storage, THIntStorage *src);
void THStorage_copyLong(THStorage *storage, THLongStorage *src);
void THStorage_copyFloat(THStorage *storage, THFloatStorage *src);
void THStorage_copyDouble(THStorage *storage, THDoubleStorage *src);
]])

local mt = torch.Storage

mt.copy =
   function(self, s)
      local stype = type(s)
      if stype == 'torch.Storage' then
         TH.THStorage_copy(self, s)
      elseif stype == 'torch.ByteStorage' then
         TH.THStorage_copyByte(self, s)
      elseif stype == 'torch.CharStorage' then
         TH.THStorage_copyChar(self, s)
      elseif stype == 'torch.ShortStorage' then
         TH.THStorage_copyShort(self, s)
      elseif stype == 'torch.IntStorage' then
         TH.THStorage_copyInt(self, s)
      elseif stype == 'torch.LongStorage' then
         TH.THStorage_copyLong(self, s)
      elseif stype == 'torch.FloatStorage' then
         TH.THStorage_copyFloat(self, s)
      elseif stype == 'torch.DoubleStorage' then
         TH.THStorage_copyDouble(self, s)
      end
      return self
   end

mt.rawCopy =
   function(self, ptr)
      TH.THStorage_rawCopy(self, ptr)
      return self
   end

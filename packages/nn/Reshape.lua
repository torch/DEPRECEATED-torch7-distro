local Reshape, parent = torch.class('nn.Reshape', 'nn.Module')

function Reshape:__init(...)
   parent.__init(self)
   self.size = torch.LongStorage()
   local n = select('#', ...)
   if n == 1 and torch.typename(select(1, ...)) == 'torch.LongStorage' then
      self.size:resize(#select(1, ...)):copy(select(1, ...))
   else
      self.size:resize(n)
      for i=1,n do
         self.size[i] = select(i, ...)
      end
   end
end

function Reshape:forward(input)
   -- infer dimensions if missing
   self.resolvedSize = self.resolvedSize or torch.LongStorage()
   self.resolvedSize:resize(#self.size)
   local next = 1
   for i = 1,#self.size do
      if self.size[i] == -1 then
         self.resolvedSize[i] = input:size(next)
         next = next + 1
      else
         self.resolvedSize[i] = self.size[i]
      end
   end

   -- reshape input with given dimensions
   input = input:contiguous()
   self.output:set(input):resize(self.resolvedSize)
   return self.output
end

function Reshape:backward(input, gradOutput)
   self.gradInput:resizeAs(input)
   return self.gradInput:copy(gradOutput)
end

function Reshape:write(file)
   parent.write(self, file)
   file:writeObject(self.size)
end

function Reshape:read(file, version)
   parent.read(self, file)
   if version > 0 then
      self.size = file:readObject()
   else
      local size = file:readObject()
      self.size = torch.LongStorage(size:size())
      self.size:copy(size)
   end
end

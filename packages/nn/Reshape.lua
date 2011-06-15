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
   self.output:resize(self.size)
end

function Reshape:forward(input)
   return self.output:copy(input)
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

local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
end

function Module:forward(input)
   return self.output
end

function Module:backward(input, gradOutput)
   return self.gradInput
end

function Module:zeroGradParameters()
end

function Module:updateParameters(learningRate)
end

function Module:write(file)
   file:writeObject(self.gradInput)
   file:writeObject(self.output)
end

function Module:read(file)
   self.gradInput = file:readObject()
   self.output = file:readObject()
end

function Module:share(mlp, ...)
   for i,v in ipairs(arg) do
      if self[v] ~=nil then self[v]:set(mlp[v]) end
   end
end

function Module:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#',...) > 0 then
      clone:share(self,...)
   end
   return clone
end

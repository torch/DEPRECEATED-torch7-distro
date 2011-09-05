local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
end

function Module:parameters()
   if self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}
   else
      return
   end
end

function Module:forward(input)
   return self.output
end

function Module:backward(input, gradOutput)
   return self.gradInput
end

function Module:accGradParameters(input, gradOutput, scale)
end

function Module:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
--    if self:parameters() then
--       self:zeroGradParameters()
--       self:backward(input, gradOutput)
--       self:accGradParameters(input, gradOutput, 1)
--       self:updateParameters(lr)
--    else
--       self:backward(input, gradOutput)
--    end
end

function Module:zeroGradParameters()
   local _,gradParams = self:parameters()
   if gradParams then
      for i=1,#gradParams do
         gradParams[i]:zero()
      end
   end
end

function Module:updateParameters(learningRate)
   local params, gradParams = self:parameters()
   if params then
      for i=1,#params do
         params[i]:add(-learningRate, gradParams[i])
      end
   end
end

function Module:write(file)
   local var = {}
   for k,v in pairs(self) do
      local tk = type(k)
      if tk == 'number'
         or tk == 'string'
         or tk == 'boolean'
         or tk == 'table'
         or (tk == 'userdata' and torch.typename(self))
      then
         var[k] = v
      end
   end
   file:writeObject(var)
end

function Module:read(file)
   local var = file:readObject(var)
   for k,v in pairs(var) do
      self[k] = v
   end
end

function Module:share(mlp, ...)
   for i,v in ipairs(arg) do
      if self[v] ~= nil then self[v]:set(mlp[v]) end
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

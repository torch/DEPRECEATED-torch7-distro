local ConcatTable, parent = torch.class('nn.ConcatTable', 'nn.Module')

function ConcatTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
end

function ConcatTable:add(module)
   table.insert(self.modules, module)
end

function ConcatTable:get(index)
   return self.modules[index]
end

function ConcatTable:size()
   return #self.modules 
end

function ConcatTable:forward(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:forward(input)
   end
   return self.output
end

function ConcatTable:backward(input, gradOutput)
   for i,module in ipairs(self.modules) do
      local currentGradInput = module:backward(input, gradOutput[i])
      if i == 1 then
         self.gradInput:resizeAs(currentGradInput):copy(currentGradInput)
      else
         self.gradInput:add(currentGradInput)
      end
   end
   return self.gradInput
end

function ConcatTable:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ConcatTable:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function ConcatTable:write(file)
   parent.write(self, file)
   file:writeObject(self.modules)
end

function ConcatTable:read(file)
   parent.read(self, file)
   self.modules = file:readObject()
end

function ConcatTable:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end



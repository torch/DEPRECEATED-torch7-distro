local ParallelTable, parent = torch.class('nn.ParallelTable', 'nn.Module')

function ParallelTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
   self.gradInput = {}
end

function ParallelTable:add(module)
   table.insert(self.modules, module)
end

function ParallelTable:get(index)
   return self.modules[index]
end

function ParallelTable:size()
   return #self.modules 
end

function ParallelTable:forward(input)
   for i=1,#self.modules do
      self.output[i] = self.modules[i]:forward(input[i])
   end
   return self.output
end


function ParallelTable:backward(input, gradOutput)
   for i,module in ipairs(self.modules) do
      self.gradInput[i]= module:backward(input[i], gradOutput[i])
   end
   return self.gradInput
end

function ParallelTable:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function ParallelTable:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function ParallelTable:write(file)
   parent.write(self, file)
   file:writeObject(self.modules)
end

function ParallelTable:read(file)
   parent.read(self, file)
   self.modules = file:readObject()
end

function ParallelTable:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end




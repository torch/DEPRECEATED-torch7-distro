local Concat, parent = torch.class('nn.Concat', 'nn.Module')

function Concat:__init(dimension)
   parent.__init(self)
   self.modules = {}
   self.size = torch.LongStorage()
   self.dimension = dimension
end

function Concat:add(module)
   table.insert(self.modules, module)
end

function Concat:get(index)
   return self.modules[index]
end

function Concat:forward(input)
   for i=1,#self.modules do
      local currentOutput = self.modules[i]:forward(input)
      
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] + currentOutput:size(self.dimension)
      end
   end
   self.output:resize(self.size)
   
   local offset = 1
   for _,module in ipairs(self.modules) do
      local currentOutput = module:forward(input)
      self.output:narrow(self.dimension, offset, currentOutput:size(self.dimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.output
end

function Concat:backward(input, gradOutput)
   self.gradInput:resizeAs(input)

   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module:backward(input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)))
        
      if i==1 then
         self.gradInput:copy(currentGradInput)
      else
         self.gradInput:add(currentGradInput)
      end
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.gradInput
end

function Concat:zeroGradParameters()
   for _,module in ipairs(self.modules) do
      module:zeroGradParameters()
   end
end

function Concat:updateParameters(learningRate)
   for _,module in ipairs(self.modules) do
      module:updateParameters(learningRate)
   end
end

function Concat:write(file)
   parent.write(self, file)
   file:writeObject(self.modules)
   file:writeObject(self.size)
   file:writeInt(self.dimension)
end

function Concat:read(file, version)
   parent.read(self, file)
   self.modules = file:readObject()
   if version > 0 then
      self.size = file:readObject()
   else
      local size = file:readObject()
      self.size = torch.LongStorage(size:size())
      self.size:copy(size)
   end
   self.dimension = file:readInt()
end


function Concat:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

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

function Concat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module:accGradParameters(input,
                                                        gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
                                                        scale)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:accUpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local currentGradInput = module:accUpdateGradParameters(input,
                                                              gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
                                                              lr)
      offset = offset + currentOutput:size(self.dimension)
   end
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

function Concat:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

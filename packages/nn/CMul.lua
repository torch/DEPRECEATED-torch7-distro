local CMul, parent = torch.class('nn.CMul', 'nn.Module')

function CMul:__init(inputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(inputSize)
   self.gradWeight = torch.Tensor(inputSize)
   
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end

 
function CMul:reset(stdv)
   self.weight:apply(function()
                                  return 1;
                        end)
end

function CMul:forward(input)
   self.output:copy(input);
   self.output:cmul(self.weight);
   return self.output
end

function CMul:backward(input, gradOutput)
   self.gradWeight:addcmul(1, input, gradOutput)
  
   self.gradInput:zero()
   self.gradInput:addcmul(1, self.weight, gradOutput)
   return self.gradInput
end

function CMul:zeroGradParameters()
   self.gradWeight:zero()
end

function CMul:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
end

function CMul:write(file)
   parent.write(self, file)
   file:writeObject(self.weight)
   file:writeObject(self.gradWeight)
end

function CMul:read(file)
   parent.read(self, file) 
   self.weight = file:readObject()
   self.gradWeight = file:readObject()
end

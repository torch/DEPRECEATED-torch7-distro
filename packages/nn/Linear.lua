local Linear, parent = torch.class('nn.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
  
   self.weight = torch.Tensor(inputSize, outputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(inputSize, outputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)

   self:reset()
end


function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   -- we do this so the initialization is exactly
   -- the same than in previous torch versions
   for i=1,self.weight:size(2) do
      self.weight:select(2, i):apply(function()
                                        return random.uniform(-stdv, stdv)
                                     end)
      self.bias[i] = random.uniform(-stdv, stdv)
   end
end

function Linear:forward(input)
   self.output:copy(self.bias)
   self.output:addT2dotT1(1, self.weight:t(), input)
   return self.output
end

function Linear:backward(input, gradOutput)
   self.gradWeight:addT1outT1(1, input, gradOutput)
   self.gradBias:add(gradOutput)
  
   if self.weightDecay ~= 0 then
      self.gradWeight:add(self.weightDecay, self.weight)
   end
   
   self.gradInput:zero()
   self.gradInput:addT2dotT1(1, self.weight, gradOutput)
   return self.gradInput
end

function Linear:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function Linear:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
   self.bias:add(-learningRate, self.gradBias)
end

function Linear:write(file)
   parent.write(self, file)
   file:writeDouble(self.weightDecay)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function Linear:read(file)
   parent.read(self, file)
   self.weightDecay = file:readDouble()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()
end

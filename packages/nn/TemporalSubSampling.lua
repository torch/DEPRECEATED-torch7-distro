local TemporalSubSampling, parent = torch.class('nn.TemporalSubSampling', 'nn.Module')

function TemporalSubSampling:__init(inputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(inputFrameSize)
   self.bias = torch.Tensor(inputFrameSize)
   self.gradWeight = torch.Tensor(inputFrameSize)
   self.gradBias = torch.Tensor(inputFrameSize)
   
   self:reset()
end

function TemporalSubSampling:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return random.uniform(-stdv, stdv)
                   end)   
end

function TemporalSubSampling:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function TemporalSubSampling:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
   self.bias:add(-learningRate, self.gradBias)
end

function TemporalSubSampling:write(file)
   parent.write(self, file)
   file:writeInt(self.kW)
   file:writeInt(self.dW)
   file:writeInt(self.inputFrameSize)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function TemporalSubSampling:read(file)
   parent.read(self, file)
   self.kW = file:readInt()
   self.dW = file:readInt()
   self.inputFrameSize = file:readInt()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()
end

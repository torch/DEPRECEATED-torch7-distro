local TemporalConvolution, parent = torch.class('nn.TemporalConvolution', 'nn.Module')

function TemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(inputFrameSize, kW, outputFrameSize)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(inputFrameSize, kW, outputFrameSize)
   self.gradBias = torch.Tensor(outputFrameSize)
   
   self:reset()
end

function TemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return random.uniform(-stdv, stdv)
                   end)   
end

function TemporalConvolution:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function TemporalConvolution:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
   self.bias:add(-learningRate, self.gradBias)
end

function TemporalConvolution:write(file)
   parent.write(self, file)
   file:writeInt(self.kW)
   file:writeInt(self.dW)
   file:writeInt(self.inputFrameSize)
   file:writeInt(self.outputFrameSize)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function TemporalConvolution:read(file)
   parent.read(self, file)
   self.kW = file:readInt()
   self.dW = file:readInt()
   self.inputFrameSize = file:readInt()
   self.outputFrameSize = file:readInt()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()
end

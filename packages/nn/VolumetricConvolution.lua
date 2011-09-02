local VolumetricConvolution, parent = torch.class('nn.VolumetricConvolution', 'nn.Module')

function VolumetricConvolution:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
   parent.__init(self)

   dT = dT or 1
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kT = kT
   self.kW = kW
   self.kH = kH
   self.dT = dT
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function VolumetricConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kT*self.kW*self.kH*self.nInputPlane)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return random.uniform(-stdv, stdv)
                   end)   
end

function VolumetricConvolution:forward(input)
   return input.nn.VolumetricConvolution_forward(self, input)
end

function VolumetricConvolution:backward(input, gradOutput)
   return input.nn.VolumetricConvolution_backward(self, input, gradOutput)
end

function VolumetricConvolution:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function VolumetricConvolution:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
   self.bias:add(-learningRate, self.gradBias)
end

function VolumetricConvolution:write(file)
   parent.write(self, file)
   file:writeInt(self.kT)
   file:writeInt(self.kW)
   file:writeInt(self.kH)
   file:writeInt(self.dT)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
   file:writeInt(self.nInputPlane)
   file:writeInt(self.nOutputPlane)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function VolumetricConvolution:read(file)
   parent.read(self, file)
   self.kT = file:readInt()
   self.kW = file:readInt()
   self.kH = file:readInt()
   self.dT = file:readInt()
   self.dW = file:readInt()
   self.dH = file:readInt()
   self.nInputPlane = file:readInt()
   self.nOutputPlane = file:readInt()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()
end

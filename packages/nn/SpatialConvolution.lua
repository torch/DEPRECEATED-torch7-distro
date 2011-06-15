local SpatialConvolution, parent = torch.class('nn.SpatialConvolution', 'nn.Module')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(kW, kH, nInputPlane, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(kW, kH, nInputPlane, nOutputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return random.uniform(-stdv, stdv)
                   end)   
end

function SpatialConvolution:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function SpatialConvolution:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
   self.bias:add(-learningRate, self.gradBias)
end

function SpatialConvolution:write(file)
   parent.write(self, file)
   file:writeInt(self.kW)
   file:writeInt(self.kH)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
   file:writeInt(self.nInputPlane)
   file:writeInt(self.nOutputPlane)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function SpatialConvolution:read(file)
   parent.read(self, file)
   self.kW = file:readInt()
   self.kH = file:readInt()
   self.dW = file:readInt()
   self.dH = file:readInt()
   self.nInputPlane = file:readInt()
   self.nOutputPlane = file:readInt()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()
end

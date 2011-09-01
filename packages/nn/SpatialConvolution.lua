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

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
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

function SpatialConvolution:forward(input)
   return input.nn.SpatialConvolution_forward(self, input)
end

function SpatialConvolution:backward(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolution_backward(self, input, gradOutput)
   end
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolution_backward(self, input, gradOutput, scale)
end

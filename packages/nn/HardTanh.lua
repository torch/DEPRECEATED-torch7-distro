local HardTanh = torch.class('nn.HardTanh', 'nn.Module')

function HardTanh:forward(input)
   return input.nn.HardTanh_forward(self, input)
end

function HardTanh:backward(input, gradOutput)
   return input.nn.HardTanh_backward(self, input, gradOutput)
end

local HardTanh = torch.class('nn.HardTanh', 'nn.Module')

function HardTanh:forward(input)
   return input.nn.HardTanh_forward(self, input)
end

function HardTanh:updateGradInput(input, gradOutput)
   return input.nn.HardTanh_updateGradInput(self, input, gradOutput)
end

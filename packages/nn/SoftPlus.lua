local SoftPlus = torch.class('nn.SoftPlus', 'nn.Module')

function SoftPlus:forward(input)
   return input.nn.SoftPlus_forward(self, input)
end

function SoftPlus:backward(input, gradOutput)
   return input.nn.SoftPlus_backward(self, input, gradOutput)
end

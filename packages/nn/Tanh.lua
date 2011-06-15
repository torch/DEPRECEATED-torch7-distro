local Tanh = torch.class('nn.Tanh', 'nn.Module')

function Tanh:forward(input)
   return input.nn.Tanh_forward(self, input)
end

function Tanh:backward(input, gradOutput)
   return input.nn.Tanh_backward(self, input, gradOutput)
end

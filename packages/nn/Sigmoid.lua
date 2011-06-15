local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:forward(input)
   return input.nn.Sigmoid_forward(self, input)
end

function Sigmoid:backward(input, gradOutput)
   return input.nn.Sigmoid_backward(self, input, gradOutput)
end

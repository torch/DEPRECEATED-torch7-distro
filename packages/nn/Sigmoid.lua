local Sigmoid = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:forward(input)
   return input.nn.Sigmoid_forward(self, input)
end

function Sigmoid:updateGradInput(input, gradOutput)
   return input.nn.Sigmoid_updateGradInput(self, input, gradOutput)
end

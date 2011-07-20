local SoftMax, parent = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:forward(input)
   return input.nn.SoftMax_forward(self, input)
end

function SoftMax:backward(input, gradOutput)
   return input.nn.SoftMax_backward(self, input, gradOutput)
end

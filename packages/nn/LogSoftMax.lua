local LogSoftMax = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:forward(input)
   return input.nn.LogSoftMax_forward(self, input)
end

function LogSoftMax:backward(input, gradOutput)
   return input.nn.LogSoftMax_backward(self, input, gradOutput)
end

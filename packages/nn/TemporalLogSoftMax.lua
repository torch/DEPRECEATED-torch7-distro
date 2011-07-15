local TemporalLogSoftMax, parent = torch.class('nn.TemporalLogSoftMax', 'nn.Module')

function TemporalLogSoftMax:forward(input)
   return input.nn.TemporalLogSoftMax_forward(self, input)
end

function TemporalLogSoftMax:backward(input, gradOutput)
   return input.nn.TemporalLogSoftMax_backward(self, input, gradOutput)
end

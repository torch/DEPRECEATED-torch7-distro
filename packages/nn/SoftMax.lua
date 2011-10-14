local SoftMax, parent = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:forward(input)
   return input.nn.SoftMax_forward(self, input)
end

function SoftMax:updateGradInput(input, gradOutput)
   return input.nn.SoftMax_updateGradInput(self, input, gradOutput)
end

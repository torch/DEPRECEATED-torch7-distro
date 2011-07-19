local HardShrink = torch.class('nn.HardShrink', 'nn.Module')

function HardShrink:forward(input)
   input.nn.HardShrink_forward(self, input)
   return self.output
end

function HardShrink:backward(input, gradOutput)
   input.nn.HardShrink_backward(self, input, gradOutput)
   return self.gradInput
end

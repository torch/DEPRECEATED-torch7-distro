local HardShrink, parent = torch.class('nn.HardShrink', 'nn.Module')

function HardShrink:__init(lam)
   parent.__init(self)
   self.lambda = lam or 0.5
end

function HardShrink:forward(input)
   input.nn.HardShrink_forward(self, input)
   return self.output
end

function HardShrink:updateGradInput(input, gradOutput)
   input.nn.HardShrink_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

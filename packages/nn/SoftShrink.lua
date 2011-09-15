local SoftShrink, parent = torch.class('nn.SoftShrink', 'nn.Module')

function SoftShrink:__init(lam)
   parent.__init(self)
   self.lambda = lam or 0.5
end

function SoftShrink:forward(input)
   input.nn.SoftShrink_forward(self, input)
   return self.output
end

function SoftShrink:backward(input, gradOutput)
   input.nn.SoftShrink_backward(self, input, gradOutput)
   return self.gradInput
end

local Abs, parent = torch.class('nn.Abs', 'nn.Module')

function Abs:__init()
   parent.__init(self)
end

function Abs:forward(input)
   input.nn.Abs_forward(self, input)
   return self.output
end

function Abs:backward(input, gradOutput)
   input.nn.Abs_backward(self, input, gradOutput)
   return self.gradInput
end

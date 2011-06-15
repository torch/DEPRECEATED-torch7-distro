local ClassNLLCriterion, parent = torch.class('nn.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init()
   parent.__init(self)
end

function ClassNLLCriterion:forward(input, target)
   self.output = -input[target]
   return self.output
end

function ClassNLLCriterion:backward(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   self.gradInput[target] = -1
   return self.gradInput
end

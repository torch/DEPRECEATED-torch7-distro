local AbsCriterion, parent = torch.class('nn.AbsCriterion', 'nn.Criterion')

function AbsCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function AbsCriterion:forward(input, target)
   return input.nn.AbsCriterion_forward(self, input, target)
end

function AbsCriterion:backward(input, target)
   return input.nn.AbsCriterion_backward(self, input, target)
end

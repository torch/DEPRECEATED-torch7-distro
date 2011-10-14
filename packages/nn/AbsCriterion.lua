local AbsCriterion, parent = torch.class('nn.AbsCriterion', 'nn.Criterion')

function AbsCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function AbsCriterion:forward(input, target)
   return input.nn.AbsCriterion_forward(self, input, target)
end

function AbsCriterion:updateGradInput(input, target)
   return input.nn.AbsCriterion_updateGradInput(self, input, target)
end

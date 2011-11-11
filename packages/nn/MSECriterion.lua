local MSECriterion, parent = torch.class('nn.MSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:forward(input, target)
   return input.nn.MSECriterion_forward(self, input, target)
end

function MSECriterion:updateGradInput(input, target)
   return input.nn.MSECriterion_updateGradInput(self, input, target)
end

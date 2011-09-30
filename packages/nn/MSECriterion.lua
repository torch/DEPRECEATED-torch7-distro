local MSECriterion, parent = torch.class('nn.MSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:forward(input, target)
   return input.nn.MSECriterion_forward(self, input, target)
end

function MSECriterion:backward(input, target)
   return input.nn.MSECriterion_backward(self, input, target)
end

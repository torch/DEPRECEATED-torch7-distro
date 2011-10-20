local MultiMarginCriterion, parent = torch.class('nn.MultiMarginCriterion', 'nn.Criterion')

function MultiMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MultiMarginCriterion:forward(input, target)
   return input.nn.MultiMarginCriterion_forward(self, input, target)
end

function MultiMarginCriterion:updateGradInput(input, target)
   return input.nn.MultiMarginCriterion_updateGradInput(self, input, target)
end

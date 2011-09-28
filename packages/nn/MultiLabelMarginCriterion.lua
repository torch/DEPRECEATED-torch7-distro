local MultiLabelMarginCriterion, parent = torch.class('nn.MultiLabelMarginCriterion', 'nn.Criterion')

function MultiLabelMarginCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MultiLabelMarginCriterion:forward(input, target)
   return input.nn.MultiLabelMarginCriterion_forward(self, input, target)
end

function MultiLabelMarginCriterion:backward(input, target)
   return input.nn.MultiLabelMarginCriterion_backward(self, input, target)
end

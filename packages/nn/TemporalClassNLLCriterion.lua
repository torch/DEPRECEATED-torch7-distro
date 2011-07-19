local TemporalClassNLLCriterion, parent = torch.class('nn.TemporalClassNLLCriterion', 'nn.Criterion')

function TemporalClassNLLCriterion:__init()
   parent.__init(self)
end

function TemporalClassNLLCriterion:forward(input, target)
   self.output = 0
   for i=1,target:size(1) do
      self.output = self.output - input[i][target[i]]
   end
   return self.output
end

function TemporalClassNLLCriterion:backward(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   for i=1,target:size(1) do
      self.gradInput[i][target[i]] = -1
   end
   return self.gradInput
end

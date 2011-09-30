local MSECriterion, parent = torch.class('nn.MSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:forward(input, target)
   if input:dim() == 1 then
      self.output = input.nn.MSECriterion_forward(self, input, target)
   elseif input:dim() == 2 then
      for i=1,target:size(1) do
         self.output = self.output + input.nn.MSECriterion_forward(self, input[i], target[i])
      end
   else
      error('matrix or vector expected')
   end
   return self.output
end

function MSECriterion:backward(input, target)
   self.gradInput:resizeAs(input)
   if input:dim() == 1 then
      input.nn.MSECriterion_backward(self, input, target, self.gradInput)
   elseif input:dim() == 2 then
      for i=1,target:size(1) do
         input.nn.MSECriterion_backward(self, input[i], target[i], self.gradInput[i])
      end
   else
      error('matrix or vector expected')
   end
   return self.gradInput
end

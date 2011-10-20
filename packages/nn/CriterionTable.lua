local CriterionTable, parent = torch.class('nn.CriterionTable', 'nn.Module')

function CriterionTable:__init(criterion)
   self.criterion = criterion
   self.gradInput = {criterion.gradInput}
end

function CriterionTable:forward(input) 
   self.output = self.criterion:forward(unpack(input))
   return self.output
end
    
function CriterionTable:updateGradInput(input, gradOutput)
  self.criterion:updateGradInput(unpack(input))
  return self.gradInput
end 

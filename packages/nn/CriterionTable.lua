local CriterionTable, parent = torch.class('nn.CriterionTable', 'nn.Module')

function CriterionTable:__init(criterion)
   self.criterion = criterion
   self.gradInput = {criterion.gradInput}
end

function CriterionTable:forward(input) 
   self.output = self.criterion:forward(unpack(input))
   return self.output
end
    
function CriterionTable:backward(input, gradOutput)
  self.criterion:backward(unpack(input))
  return self.gradInput
end 
 
function CriterionTable:zeroGradParameters()
  -- nothing to do: a criterion does not have any parameter
end

function CriterionTable:updateParameters(learningRate)
  -- nothing to do: a criterion does not have any parameter
end

function CriterionTable:write(file)
   parent.write(self, file)
   file:writeObject(self.criterion)
end

function CriterionTable:read(file)
   parent.read(self, file)
   self.criterion = file:readObject()
end

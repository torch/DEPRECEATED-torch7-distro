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

function AbsCriterion:write(file)
   parent.write(self, file)
   file:writeBool(self.sizeAverage)
end

function AbsCriterion:read(file)
   parent.read(self, file)
   self.sizeAverage = file:readBool()
end

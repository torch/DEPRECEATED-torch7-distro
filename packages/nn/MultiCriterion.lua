local MultiCriterion, parent = torch.class('nn.MultiCriterion', 'nn.Criterion')

function MultiCriterion:__init()
   parent.__init(self)
   self.criterions = {}
   self.weights = torch.DoubleStorage()
end

function MultiCriterion:add(criterion, weight)
   weight = weight or 1
   table.insert(self.criterions, criterion)
   self.weights:resize(#self.criterions, true)
   self.weights[#self.criterions] = weight
end

function MultiCriterion:forward(input, target)
   self.output = 0
   for i=1,#self.criterions do
      self.output = self.output + self.weights[i]*self.criterions[i]:forward(input, target)
   end
   return self.output
end

function MultiCriterion:backward(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   for i=1,#self.criterions do
      self.gradInput:add(self.weights[i], self.criterions[i]:backward(input, target))
   end
   return self.gradInput
end

function MultiCriterion:write(file)
   parent.write(self, file)
   file:writeObject(self.criterions)
   file:writeObject(self.weights)
end

function MultiCriterion:read(file)
   parent.read(self, file)
   self.criterions = file:readObject()
   self.weights = file:readObject()
end

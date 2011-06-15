local MSECriterion, parent = torch.class('nn.MSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:write(file)
   parent.write(self, file)
   file:writeBool(self.sizeAverage)
end

function MSECriterion:read(file)
   parent.read(self, file)
   self.sizeAverage = file:readBool()
end
